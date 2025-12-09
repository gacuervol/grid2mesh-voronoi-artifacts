import numpy as np
import xarray as xr
import pytest
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from download_data import generate_sea_depth_bathymetry_from_etopo_rescaled_and_filtered

def dummy_generate_sea_depth_topography_from_etopo(url):
    """
    Returns a Dataset with a 4x4 'elevation' array containing negative values,
    including 'lat' and 'lon' coordinates.
    """
    data = np.array([
        [-10, -20, -30, -40],
        [-50, -60, -70, -80],
        [-90, -100, -110, -120],
        [-130, -140, -150, -160]
    ])
    # Create latitude and longitude coordinates
    lat = np.linspace(-10, 10, 4)
    lon = np.linspace(-20, 20, 4)
    ds = xr.Dataset(
        {"elevation": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon}
    )
    return ds

def dummy_resize_lonxlat(dataset: xr.Dataset, new_lonxlat: tuple) -> xr.Dataset:
    """
    Dummy function that simulates resizing the longitude and latitude dimensions of an xarray dataset.
    
    Parameters:
    dataset (xr.Dataset): The input xarray dataset.
    new_lonxlat (tuple): A tuple containing the new size of longitude and latitude dimensions, e.g., (lon_size, lat_size).

    Returns:
    xr.Dataset: A dummy resized xarray dataset.
    
    Raises:
    ValueError: If the dataset does not contain longitude and latitude coordinates.
    """
    # Find longitude and latitude coordinate names
    lon_keys = [key for key in dataset.coords.keys() if key.startswith("lon")]
    lat_keys = [key for key in dataset.coords.keys() if key.startswith("lat")]
    if not lon_keys or not lat_keys:
        raise ValueError("Dataset does not contain longitude and latitude coordinates")
    lon_name = lon_keys[0]
    lat_name = lat_keys[0]

    # Get start and end values of longitude and latitude
    lon_start, lon_end = dataset[lon_name].values[0], dataset[lon_name].values[-1]
    lat_start, lat_end = dataset[lat_name].values[0], dataset[lat_name].values[-1]

    # Extract new sizes of longitude and latitude
    lon_size, lat_size = new_lonxlat

    # Create new coordinate values using linspace
    new_lon = np.linspace(lon_start, lon_end, lon_size)
    new_lat = np.linspace(lat_start, lat_end, lat_size)

    # Simulate resizing by filling the new array with the mean value of the original 'elevation' data.
    original_data = dataset["elevation"].values
    dummy_value = original_data.mean()
    new_data = np.full((lat_size, lon_size), dummy_value)

    # Create a new dataset with the new coordinates and dummy data
    ds_dummy = xr.Dataset(
        {"elevation": (("lat", "lon"), new_data)},
        coords={lat_name: new_lat, lon_name: new_lon}
    )
    return ds_dummy

def dummy_select_ocean_region(ds: xr.Dataset, sea_mask: np.ndarray) -> xr.Dataset:
    """
    Applies the sea mask: where sea_mask is False, assign 0.
    """
    ds["elevation"] = ds["elevation"].where(sea_mask, other=0)
    return ds

def test_generate_sea_depth_bathymetry(monkeypatch):
    # Replace external functions with dummy versions in order to control the input and output values.
    monkeypatch.setattr(
        "download_data.generate_sea_depth_topography_from_etopo",
        dummy_generate_sea_depth_topography_from_etopo
    )
    monkeypatch.setattr(
        "download_data.resize_lonxlat",
        dummy_resize_lonxlat
    )
    monkeypatch.setattr(
        "download_data.select_ocean_region",
        dummy_select_ocean_region
    )
    
    # Input parameters for the function tested:
    url = "http://example.com/dummy"
    # We want to downsample a 4x4 dataset to 2x2
    resolution = (2, 2)
    # Define a sea mask for the 2x2 dataset where True indicates ocean and False indicates land.
    sea_mask = np.array([[True, False],
                         [False, True]])
    
    result = generate_sea_depth_bathymetry_from_etopo_rescaled_and_filtered(url, resolution, sea_mask)
    
    # Expected flow:
    # 1. dummy_generate_sea_depth_topography_from_etopo returns a 4x4 'elevation' dataset.
    # 2. dummy_resize_lonxlat performs resizing:
    #    It simulates interpolation by filling the new 2x2 array with the mean of the original data.
    #    Mean of original 'elevation' data:
    #      (-10-20-30-40-50-60-70-80-90-100-110-120-130-140-150-160) / 16 = -85
    # 3. The main function then multiplies 'elevation' by -1: resulting in 85.
    # 4. dummy_select_ocean_region applies the sea mask:
    #    With the mask [[True, False], [False, True]], the expected result is:
    #      [[85, 0],
    #       [0, 85]]
    expected = np.array([[85, 0],
                         [0, 85]])
    
    # Verify that the result is as expected
    np.testing.assert_array_equal(result["elevation"].values, expected)
