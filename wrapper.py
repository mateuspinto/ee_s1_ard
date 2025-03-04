#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version: v1.2
Date: 2021-04-01
Authors: Mullissa A., Vollrath A., Braun, C., Slagter B., Balling J., Gou Y., Gorelick N.,  Reiche J.
Description: A wrapper function to derive the Sentinel-1 ARD
"""

from typing import Optional
import ee
import ee_s1_ard.border_noise_correction as bnc
import ee_s1_ard.speckle_filter as sf
import ee_s1_ard.terrain_flattening as trf
import ee_s1_ard.helper as helper


###########################################
# DO THE JOB
###########################################


class S1ARDImageCollection:
    """
    Creates an Analysis-Ready Sentinel-1 ImageCollection following the framework
    described in Mullissa et al. (2021) Remote Sens. 2021, 13, 1954.

    Parameters
    ----------
    geometry : ee.Geometry
        Region of interest for filtering the Sentinel-1 collection.
    start_date : str
        Start date for filtering the Sentinel-1 collection (YYYY-MM-DD).
        See Section 2.1 (Mullissa et al. 2021) for details on data selection.
    stop_date : str
        End date for filtering the Sentinel-1 collection (YYYY-MM-DD).
    ascending: bool, default=True
        Choose between ascending or descending orbit.
    polarization : str, default="VVVH"
        Polarization mode. Options are "VV", "VH", "VVVH".
        See Section 2.1 for coverage details.
    apply_border_noise_correction : bool, default=True
        Whether to remove additional border noise. See Section 2.2.
    apply_terrain_flattening : bool, default=True
        Whether to apply radiometric terrain normalization. See Section 2.4.
    apply_speckle_filtering : bool, default=True
        Whether to apply speckle filtering. See Section 2.3.
    speckle_filter_framework : str, default="MULTI"
        Framework for speckle filtering. Options are "MONO" or "MULTI".
    speckle_filter_type : str, default="GAMMA MAP"
        Type of speckle filter. Options include "BOXCAR", "LEE",
        "GAMMA MAP", "REFINED LEE", "LEE SIGMA".
    speckle_filter_kernel_size : int, default=7
        Kernel size used by the speckle filter.
    speckle_filter_nr_of_images : int, default=10
        Number of images considered in multi-temporal speckle filtering.
    terrain_flattening_model : str, default="VOLUME"
        Model for terrain normalization. Options are "DIRECT", "VOLUME".
    terrain_flattening_additional_buffer : int, default=0
        Additional buffer in meters for layover and shadow masking.
    output_format : str, default="DB"
        Output backscatter format. Options are "LINEAR" or "DB".
    dem : Optional[ee.Image], default=None
        Digital Elevation Model for terrain normalization. If None,
        the default SRTM DEM is used.

    References
    ----------
    Mullissa, A. et al. (2021): Sentinel-1 SAR Backscatter Analysis Ready Data Preparation
    in Google Earth Engine. Remote Sens. 13(10), 1954.
    """

    def __init__(
        self,
        geometry: ee.Geometry,
        start_date: str,
        stop_date: str,
        ascending: bool = True,
        polarization: str = "VVVH",
        apply_border_noise_correction: bool = True,
        apply_terrain_flattening: bool = True,
        apply_speckle_filtering: bool = True,
        speckle_filter_framework: str = "MULTI",
        speckle_filter_type: str = "GAMMA MAP",
        speckle_filter_kernel_size: int = 7,
        speckle_filter_nr_of_images: int = 10,
        terrain_flattening_model: str = "VOLUME",
        terrain_flattening_additional_buffer: int = 0,
        output_format: str = "DB",
        clip_to_roi: bool = False,
        save_asset: bool = False,
        asset_id: str = "",
        dem: str = "USGS/SRTMGL1_003",
    ) -> None:
        if polarization not in ["VV", "VH", "VVVH"]:
            raise ValueError("Invalid polarization")
        if speckle_filter_framework not in ["MONO", "MULTI"]:
            raise ValueError("Invalid speckle filter framework")
        if speckle_filter_type not in [
            "BOXCAR",
            "LEE",
            "GAMMA MAP",
            "REFINED LEE",
            "LEE SIGMA",
        ]:
            raise ValueError("Invalid speckle filter type")
        if terrain_flattening_model not in ["DIRECT", "VOLUME"]:
            raise ValueError("Invalid terrain flattening model")
        if output_format not in ["LINEAR", "DB"]:
            raise ValueError("Invalid output format")
        if terrain_flattening_additional_buffer < 0:
            raise ValueError("Buffer must be >= 0")
        if speckle_filter_kernel_size <= 0:
            raise ValueError("Kernel size must be > 0")

        self.start_date = start_date
        self.stop_date = stop_date
        self.geometry = geometry
        self.polarization = polarization
        self.ascending = ascending
        self.apply_border_noise_correction = apply_border_noise_correction
        self.apply_terrain_flattening = apply_terrain_flattening
        self.apply_speckle_filtering = apply_speckle_filtering
        self.speckle_filter_framework = speckle_filter_framework
        self.speckle_filter_type = speckle_filter_type
        self.speckle_filter_kernel_size = speckle_filter_kernel_size
        self.speckle_filter_nr_of_images = speckle_filter_nr_of_images
        self.terrain_flattening_model = terrain_flattening_model
        self.terrain_flattening_additional_buffer = terrain_flattening_additional_buffer
        self.output_format = output_format
        self.clip_to_roi = clip_to_roi
        self.save_asset = save_asset
        self.asset_id = asset_id
        self.dem = ee.Image(dem)

    def get_collection(self) -> ee.ImageCollection:
        """
        Prepares the Sentinel-1 SAR ImageCollection as Analysis-Ready Data (ARD).

        Returns
        -------
        ee.ImageCollection
            Sentinel-1 ARD collection in linear or dB scale.
        """
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filterDate(self.start_date, self.stop_date)
            .filterBounds(self.geometry)
        )
        if self.polarization == "VV":
            s1 = s1.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
            )
            s1 = s1.select(["VV", "angle"])
        elif self.polarization == "VH":
            s1 = s1.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", "VH")
            )
            s1 = s1.select(["VH", "angle"])
        else:
            s1 = s1.filter(
                ee.Filter.And(
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
                )
            )
            s1 = s1.select(["VV", "VH", "angle"])

        s1 = s1.filter(
            ee.Filter.eq(
                "orbitProperties_pass", "ASCENDING" if self.ascending else "DESCENDING"
            )
        )

        if self.apply_border_noise_correction:
            s1 = s1.map(bnc.f_mask_edges)
        if self.apply_speckle_filtering:
            if self.speckle_filter_framework == "MONO":
                s1 = ee.ImageCollection(
                    sf.MonoTemporal_Filter(
                        s1, self.speckle_filter_kernel_size, self.speckle_filter_type
                    )
                )
            else:
                s1 = ee.ImageCollection(
                    sf.MultiTemporal_Filter(
                        s1,
                        self.speckle_filter_kernel_size,
                        self.speckle_filter_type,
                        self.speckle_filter_nr_of_images,
                    )
                )
        if self.apply_terrain_flattening:
            s1 = trf.slope_correction(
                s1,
                self.terrain_flattening_model,
                self.dem,
                self.terrain_flattening_additional_buffer,
            )
        if self.output_format == "DB":
            s1 = s1.map(helper.lin_to_db)

        return s1
