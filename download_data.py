# Standard library
import argparse
import calendar
import os
from datetime import datetime, timedelta

# Third-party
import cdsapi
import copernicusmarine as cm
import ecmwf.opendata as eo
import numpy as np
import pandas as pd
import xarray as xr
import requests
from PIL import Image
from io import BytesIO
import re

# First-party
from neural_lam import constants


def load_mask(path):
    """
    Load bathymetry mask.

    Args:
    path (str): Path to the bathymetry mask file.

    Returns:
    mask (xarray.Dataset): Bathymetry mask.
    """
    if os.path.exists(path):
        print("Bathymetry mask file found. Loading from file.")
        mask = xr.load_dataset(path).mask
    else:
        print("Bathymetry mask file not found. Downloading...")
        bathy_data = cm.open_dataset(
            dataset_id="cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE",
            variables=["analysed_sst"],
            minimum_longitude=-20.97,
            maximum_longitude=-5.975,
            minimum_latitude=19.55,
            maximum_latitude=34.525,
            start_datetime="2023-12-30T00:00:00",
            end_datetime="2023-12-31T00:00:00",
        )
        sst = bathy_data["analysed_sst"]
        mask = xr.where(np.isnan(sst.isel(time=0)), 0, 1)
        mask.name = "mask"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        mask.to_netcdf(path)
        print(f"mask saved in: {path}")

    return mask


def select(dataset, mask):
    """
    Select masked volume.

    Args:
    dataset (xarray.Dataset): Input dataset.
    mask (xarray.Dataset): Bathymetry mask.

    Returns:
    mask (xarray.Dataset): Masked dataset.
    """
    # Fix longitude mismatches close to zero
    dataset["longitude"] = dataset.longitude.where(
        (dataset.longitude < -1e-9) | (dataset.longitude > 1e-9), other=0.0
    )

    # Select depth levels, and masked area
    if hasattr(dataset, "depth"):
        dataset = dataset.sel(depth=constants.DEPTHS)
        dataset = dataset.where(mask)
    else:
        if "depth" in mask.dims:
            dataset = dataset.where(mask.isel(depth=0))
        else:
            dataset = dataset.where(mask)

    # Uncomment to coarsen grid
    # dataset = dataset.coarsen(longitude=2, boundary="pad").mean()
    # dataset = dataset.coarsen(latitude=2, boundary="pad").mean()
    return dataset


def download_static_mediterranean(path_prefix, mask):
    """
    Download static data as numpy files.

    Args:
    path_prefix (str): Path to store.
    mask (xarray.Dataset): Bathymetry mask.
    """
    # Download ocean depth and mask
    bathy_data = cm.open_dataset(
        dataset_id="cmems_mod_med_phy_my_4.2km_static",
        dataset_part="bathy",
        dataset_version="202211",
        service="static-arco",
        variables=["deptho", "mask"],
        minimum_depth=constants.DEPTHS[0],
        maximum_depth=constants.DEPTHS[-1],
    )
    bathy_data = select(bathy_data, mask)

    sea_depth = bathy_data.deptho.isel(depth=0)
    np.save(f"{path_prefix}/sea_depth.npy", np.nan_to_num(sea_depth, nan=0.0))

    # Store bathymetry mask
    sea_mask = ~np.isnan(bathy_data.mask)
    np.save(f"{path_prefix}/sea_mask.npy", sea_mask)

    # Forcing mask for the Strait of Gibraltar
    border_mask = np.where(
        bathy_data.mask.longitude < -5.2, bathy_data.mask, np.nan
    )
    border_mask = ~np.isnan(border_mask)
    np.save(f"{path_prefix}/boundary_mask.npy", border_mask)

    y_indices, x_indices = np.indices(sea_mask.shape[1:])
    nwp_xy = np.stack([x_indices, y_indices])
    np.save(f"{path_prefix}/nwp_xy.npy", nwp_xy)

    lat = bathy_data.latitude
    lon = bathy_data.longitude
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    coordinates = np.stack([lat_mesh, lon_mesh])  # (2, h, w)
    np.save(f"{path_prefix}/coordinates.npy", coordinates)

    # Download mean dynamic topography
    mdt_data = cm.open_dataset(
        dataset_id="cmems_mod_med_phy_my_4.2km_static",
        dataset_part="mdt",
        dataset_version="202211",
        service="static-arco",
        variables=["mdt"],
    )
    mdt_data = select(mdt_data, mask)
    np.save(
        f"{path_prefix}/sea_topography.npy",
        np.nan_to_num(mdt_data.mdt, nan=0.0),
    )

    # Download coordinate data
    coord_data = cm.open_dataset(
        dataset_id="cmems_mod_med_phy_my_4.2km_static",
        dataset_part="coords",
        dataset_version="202211",
        service="static-arco",
        variables=["e1t"],
    )
    coord_data = select(coord_data, mask)
    grid_weights = coord_data.e1t / coord_data.e1t.mean()
    np.save(
        f"{path_prefix}/grid_weights.npy", np.nan_to_num(grid_weights, nan=0.0)
    )

def generate_grid_weights_from_latitude_cosine(mask: xr.DataArray) -> np.ndarray:
    """
    From a 0-1 mask indicating the presence of ocean in each cell, generates weights for each cell based on latitude.
    Weights are calculated as the cosine of the latitude, normalized with respect to the mean of the weights.

    Args:
    mask: xr.DataArray containing the ocean mask. It must be 2D. It is handled within the function in case it's 3D.

    Returns:
    np.ndarray with normalized weights in ocean positions of the mask.
    """
    # Reshape the mask to ensure it is 2D and has no depth
    if len(mask.values.shape) == 3:
        mask = mask.isel(depth=0)
        
    # Create a grid of zeros with the same shape as the mask to place weights in ocean positions later
    grid_out = np.zeros(mask.values.shape)
    
    # Convert latitude coordinate to a 2D matrix matching the mask shape to select latitudes corresponding to the ocean
    lat_2d = np.broadcast_to(mask.latitude.values[:, None], mask.values.shape)

    # Extract latitudes in positions where the mask is 1, corresponding to the ocean in our case
    latitude_ocean = lat_2d[mask.values == 1]

    # Apply the formula
    grid_weights = np.cos(np.deg2rad(latitude_ocean))
    mean = np.mean(grid_weights, axis=0)
    grid_weights_normalized = grid_weights / mean

    # Place normalized grid weights in mask positions where the value is 1
    grid_out[mask.values == 1] = grid_weights_normalized
    return grid_out

def generate_sea_depth_topography_from_etopo(url : str) -> xr.Dataset: 
    """
    Generate from download url the sea depth topography data from previously selected region.
    The data is stored in a xarray dataset with the following dimensions:   
    - lat: latitude
    - lon: longitude
    - elevation: elevation in meters

    Parameters
    ----------
    url : str
        The url to download the sea depth topography data from.

    Returns
    -------
    sea_depth_topography : xarray.Dataset
        The sea depth topography data stored in a xarray dataset.    
    """

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)

    sea_depth_bathymetry = xr.Dataset(
    {
        "elevation": (("lat", "lon"), np.flipud(img))
    },
    coords={
        "lat": np.linspace(19.55, 34.525, img.shape[0]),
        "lon": np.linspace(-20.975, -5.975, img.shape[1]),
    },
)

    return sea_depth_bathymetry

def resize_lonxlat(dataset: xr.Dataset, new_lonxlat: tuple) -> xr.Dataset:
    """
    Resize the longitude and latitude dimensions of an xarray dataset.

    Parameters:
    dataset (xr.Dataset): The input xarray dataset.
    new_lonxlat (tuple): A tuple containing the new size of longitude and latitude dimensions, e.g., (lon_size, lat_size).

    Returns:
    xr.Dataset: The resized xarray dataset.

    Raises:
    ValueError: If the dataset does not contain longitude and latitude coordinates.

    Note:
    - The function interpolates the dataset to match the specified new size of the longitude and latitude dimensions.
    - When NaN values are present in the dataset, linear interpolation is used; otherwise, cubic interpolation is used.
    """
    # Find longitude and latitude coordinate names
    lon_name = [key for key in dataset.coords.keys() if re.match(r'^lon', key)][0]
    lat_name = [key for key in dataset.coords.keys() if re.match(r'^lat', key)][0]

    # Get start and end values of longitude and latitude
    lon_start, lon_end = dataset[lon_name].data[0], dataset[lon_name].data[-1]
    lat_start, lat_end = dataset[lat_name].data[0], dataset[lat_name].data[-1]

    # Extract new sizes of longitude and latitude
    lon_size, lat_size = new_lonxlat

    # Check if there are NaN values in the dataset
    there_nan = bool(dataset.isnull().any().to_array().data.any())

    # Interpolate the dataset based on whether NaN values are present
    if not there_nan:
        # Cubic interpolation when no NaN values are present
        return dataset.interp({lon_name: np.linspace(lon_start, lon_end, lon_size),
                               lat_name: np.linspace(lat_start, lat_end, lat_size)},
                               method="cubic")
    else:
        # Linear interpolation when NaN values are present
        return dataset.interp({lon_name: np.linspace(lon_start, lon_end, lon_size),
                               lat_name: np.linspace(lat_start, lat_end, lat_size)},
                               method="linear")
    
def select_ocean_region(dataset: xr.Dataset, sea_mask: np.ndarray) -> xr.Dataset:
    """
    Selecciona la región oceánica de un dataset de xarray basándose en una máscara marina.

    Parámetros:
        dataset (xr.Dataset): El dataset de entrada.
        sea_mask (np.ndarray): Un array 2D (o con una dimensión extra, por ejemplo, (1, lat, lon)) 
                               que contiene la máscara marina, donde True indica océano y False tierra.

    Retorna:
        xr.Dataset: El dataset de xarray que contiene solo la región oceánica, 
                    con los valores de tierra reemplazados por 0.

    Raises:
        ValueError: Si la máscara no tiene la misma forma que las dimensiones espaciales del dataset.
    """
    # Si la máscara tiene una dimensión extra (por ejemplo, de forma (1, lat, lon)), la reducimos a 2D
    if sea_mask.ndim == 3 and sea_mask.shape[0] == 1:
        sea_mask = np.squeeze(sea_mask, axis=0)
    
    # Verifica que la máscara tenga la misma forma que las dimensiones espaciales del dataset
    if dataset.sizes['lat'] != sea_mask.shape[0] or dataset.sizes['lon'] != sea_mask.shape[1]:
        raise ValueError("La máscara marina no tiene la misma forma que las dimensiones espaciales del dataset.")
    
    # Selecciona la región oceánica: donde la máscara es True se mantienen los datos; donde es False se asigna NaN,
    # y luego se reemplazan los NaN por 0.
    dataset_ocean = dataset.where(sea_mask).fillna(0)
    return dataset_ocean

def generate_sea_depth_bathymetry_from_etopo_rescaled_and_filtered(url : str, resolution : tuple, sea_mask : np.ndarray) -> xr.Dataset:
    """
    Generate the sea depth bathymetry data from the ETOPO1 dataset, rescaled to the specified resolution and
    filtered by the sea mask.

    Parameters:
    url (str): The URL to download the sea depth bathymetry data from.
    resolution (tuple): A tuple containing the new size of longitude and latitude dimensions, e.g., (lon_size, lat_size).
    sea_mask (np.ndarray): A 2D array that contains the sea mask, where True indicates ocean and False land.

    Returns:
    xr.Dataset: The sea depth bathymetry data stored in a xarray dataset, rescaled and filtered by the sea mask.
    """

    sea_depth_bathymetry = generate_sea_depth_topography_from_etopo(url)
    sea_depth_bathymetry_resized = resize_lonxlat(sea_depth_bathymetry, resolution)
    sea_depth_bathymetry_resized['elevation'] = -sea_depth_bathymetry_resized['elevation']
    sea_depth_bathymetry_ocean = select_ocean_region(sea_depth_bathymetry_resized, sea_mask)
    sea_depth_bathymetry_ocean["elevation"] = sea_depth_bathymetry_ocean["elevation"].where(
        sea_depth_bathymetry_ocean["elevation"] >= 0, 0
    )
    return sea_depth_bathymetry_ocean

def download_static_atlantic(
    path_prefix: str,
    username: str,
    password: str,
    ):
    """
    Function to generate static files for the Atlantic dataset (SST).
    The SST product is used to create the mask and 'dummy' files are generated
    for data that is not available (for example, depth and topography).
    """
    ds = cm.open_dataset(
        username=username,
        password=password,
        dataset_id="cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE",
        variables=["analysed_sst"],
        minimum_longitude=-20.97,
        maximum_longitude=-5.975,
        minimum_latitude=19.55,
        maximum_latitude=34.525,
        start_datetime="2023-12-30T00:00:00",
        end_datetime="2023-12-31T00:00:00",
    )
    sst = ds["analysed_sst"]

    # Create mask: it is assumed that where sst is NaN is land, where it has a value is sea.
    mask = xr.where(np.isnan(sst.isel(time=0)), 0, 1)
    mask.name = "mask"
    out_mask_path = os.path.join(path_prefix, "bathy_mask.nc")
    mask.to_netcdf(out_mask_path)
    print(f"Mask saved in {out_mask_path}")

    # Convert mask to numpy and ensure that it has a depth dimension
    mask_np = mask.values
    if mask_np.ndim == 2:
        mask_np = mask_np[np.newaxis, ...]
    np.save(os.path.join(path_prefix, "sea_mask.npy"), mask_np.astype(bool))

    # Create sea depth from etopo data
    ruta = r'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage?bbox=-20.97500,19.55000,-5.97500,34.52500&bboxSR=4326&size=1800,1797&imageSR=4326&format=tiff&pixelType=F32&interpolation=+RSP_NearestNeighbor&compression=LZ77&renderingRule={%22rasterFunction%22:%22none%22}&mosaicRule={%22where%22:%22Name=%27ETOPO_2022_v1_30s_bed%27%22}&f=image'
    resolution = mask_np.astype(bool)[0].shape
    sea_depth = generate_sea_depth_bathymetry_from_etopo_rescaled_and_filtered(ruta, resolution, mask_np.astype(bool)[0])
    np.save(os.path.join(path_prefix, "sea_depth.npy"), sea_depth['elevation'].values)

    # Sea topography: not available, create arrays of zeros
    sea_topography = np.zeros(mask_np.shape[1:])
    np.save(os.path.join(path_prefix, "sea_topography.npy"), sea_topography)

    # Boundary mask: define margins to identify edges
    lat = ds.latitude.values
    lon = ds.longitude.values
    lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
    margin_lat = 0.00 * (lat.max() - lat.min())
    margin_lon = 0.00 * (lon.max() - lon.min())
    boundary_mask = np.zeros(mask_np.shape[1:], dtype=np.int32)
    boundary_mask[(lat_mesh < lat.min() + margin_lat) |
                  (lat_mesh > lat.max() - margin_lat) |
                  (lon_mesh < lon.min() + margin_lon) |
                  (lon_mesh > lon.max() - margin_lon)] = 1
    np.save(os.path.join(path_prefix, "boundary_mask.npy"), boundary_mask)

    # nwp_xy: grid indices
    y_indices, x_indices = np.indices(mask_np.shape[1:])
    nwp_xy = np.stack([x_indices, y_indices])
    np.save(os.path.join(path_prefix, "nwp_xy.npy"), nwp_xy)

    # Coordinates: from lat and lon
    coordinates = np.stack([lat_mesh, lon_mesh])
    np.save(os.path.join(path_prefix, "coordinates.npy"), coordinates)

    # Grid weights: we assign ones
    #grid_weights = np.ones(mask_np.shape[1:])
    #np.save(os.path.join(path_prefix, "grid_weights.npy"), grid_weights)
    grid_weigths_from_latitude_cosine = generate_grid_weights_from_latitude_cosine(mask)
    np.save(os.path.join(path_prefix, "grid_weights.npy"), grid_weigths_from_latitude_cosine)
    
    ds.close()


def download_data(
    start_date,
    end_date,
    datasets,
    version,
    static_path,
    path_prefix,
    mask,
    username,
    password,
):
    """
    Download and save daily physics data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    datasets (dict): Datasets to download.
    version (str): Dataset version.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask
    """

    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    current_date = start_date
    while current_date <= end_date:
        # Calculate the first and last day of the current month
        first_day = current_date.replace(day=1)
        last_day = current_date.replace(
            day=calendar.monthrange(current_date.year, current_date.month)[1]
        )

        # Format the start and end datetime strings for the month
        start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
        end_datetime = last_day.strftime("%Y-%m-%dT00:00:00")

        # Request data for the whole month
        month_data = {}
        for dataset_id, variables in datasets.items():
            ds = cm.open_dataset(
                username=username,
                password=password,
                dataset_id=dataset_id,
                dataset_version=version,
                dataset_part="default",
                service="arco-geo-series",
                variables=variables,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                minimum_depth=constants.DEPTHS[0],
                maximum_depth=constants.DEPTHS[-1],
            )
            month_data[dataset_id] = select(ds, mask)
            ds.close()

        # Process and save daily data
        for day in range((last_day - first_day).days + 1):
            day_date = first_day + timedelta(days=day)
            filename = f"{path_prefix}/{day_date.strftime('%Y%m%d')}.npy"
            all_data = []
            for dataset_id in datasets:
                dataset = month_data[dataset_id]
                for var in datasets[dataset_id]:
                    daily_data = dataset[var].isel(time=day).values
                    if var == "bottomT":
                        daily_data = daily_data[:, :, 0]
                    if len(daily_data.shape) == 2:
                        daily_data = daily_data[np.newaxis, ...]
                    daily_data = daily_data.transpose(1, 2, 0)  # h, w, f
                    all_data.append(daily_data)

            # Concatenate all data along a new axis
            combined_data = np.concatenate(all_data, axis=-1)
            clean_data = np.nan_to_num(combined_data, nan=0.0)

            # Select only sea grid points
            sea_data = clean_data[grid_mask, :]  # n_grid, f

            # Save combined data as one numpy file
            np.save(filename, sea_data)
            print(f"Saved data to {filename}")

        # Increment the next month
        current_date = last_day + timedelta(days=1)


def download_forecast(
    start_date,
    end_date,
    datasets,
    version,
    static_path,
    path_prefix,
    mask,
):
    """
    Download and save forecast data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    datasets (dict): Datasets to download.
    version (str): Dataset version.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    filename = f"{path_prefix}/{start_date.strftime('%Y%m%d')}.npy"

    initial_date = start_date - timedelta(days=2)

    all_data = []
    for dataset_id, variables in datasets.items():
        # Load ocean physics dataset for all dates at once
        dataset = cm.open_dataset(
            dataset_id=dataset_id,
            dataset_version=version,
            dataset_part="default",
            service="arco-geo-series",
            variables=variables,
            start_datetime=initial_date.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=end_date.strftime("%Y-%m-%dT00:00:00"),
            minimum_depth=constants.DEPTHS[0],
            maximum_depth=constants.DEPTHS[-1],
        )

        dataset = select(dataset, mask)
        for var in variables:
            data = dataset[var].values
            if var == "bottomT":
                data = data[:, :, :, 0]
            if len(data.shape) == 3:
                data = data[:, np.newaxis, ...]  # t, 1, h, w
            data = data.transpose(0, 2, 3, 1)  # t, h, w, f
            all_data.append(data)

        dataset.close()

    # Concatenate all data along the feature dimension
    combined_data = np.concatenate(all_data, axis=-1)
    combined_data = np.nan_to_num(combined_data, nan=0.0)

    # Select only sea grid points
    sea_data = combined_data[:, grid_mask, :]  # t, n_grid, f

    # Save combined data as one numpy file
    np.save(filename, sea_data)
    print(f"Saved forecast data to {filename}")


def download_era5(
    start_date,
    end_date,
    request_variables,
    ds_variables,
    static_path,
    path_prefix,
    mask,
):
    """
    Download and save daily ERA5 data.

    Args:
    start_date (datetime): The start date for data retrieval.
    end_date (datetime): The end date for data retrieval.
    request_variables (list): List of variables to request from cds.
    ds_variables (list): List of variables in the dataset.
    static_path (str): Location of static data.
    path_prefix (str): The directory path prefix where the files will be saved.
    mask (xarray.Dataset): Bathymetry mask.
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]
    client = cdsapi.Client()
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month

        filename = f"{path_prefix}/{current_date.strftime('%Y%m')}.nc"
        if os.path.isfile(filename):
            if month == 7:
                current_date = current_date.replace(
                    year=year + 1, month=1, day=1
                )
            else:
                current_date = current_date.replace(month=month + 6, day=1)
            continue

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "format": "netcdf",
                "product_type": ["reanalysis"],
                "variable": request_variables,
                "year": [str(year)],
                "month": [f"{m:02d}" for m in range(month, month + 6)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                "area": [
                    mask.latitude.max().item(),
                    mask.longitude.min().item(),
                    mask.latitude.min().item(),
                    mask.longitude.max().item(),
                ],
                "download_format": "unarchived",
            },
            filename,
        )
        print(f"Downloaded {filename}")

        # Load the data and average to daily
        ds = xr.open_dataset(filename)
        daily_ds = ds.resample(valid_time="1D").mean()

        # Interpolate to the bathymetry mask grid
        interp_daily_ds = daily_ds.interp(
            longitude=mask.longitude, latitude=mask.latitude
        )

        # Apply the bathymetry mask to select the exact area
        masked_data = select(interp_daily_ds, mask)

        # Save in numpy format with shape (n_grid, f)
        for single_date in masked_data.valid_time.values:
            date_str = pd.to_datetime(single_date).strftime("%Y%m%d")
            daily_data = masked_data.sel(valid_time=single_date)
            combined_data = []
            for var in ds_variables:
                data = daily_data[var].values  # h, w
                combined_data.append(data)

            combined_data = np.stack(combined_data, axis=-1)  # h, w, f
            clean_data = np.nan_to_num(combined_data, nan=0.0)

            np.save(
                f"{path_prefix}/{date_str}.npy", clean_data[grid_mask, :]
            )  # n_grid, f
            print(f"Saved daily data to {path_prefix}/{date_str}.npy")

        # Increment to the next month
        if month == 7:
            current_date = current_date.replace(year=year + 1, month=1, day=1)
        else:
            current_date = current_date.replace(month=month + 6, day=1)



def download_ecmwf_forecast(
    start_date,
    static_path,
    mask,
    request_variables,
    ds_variables,
    max_step,
    model,
    product,
    path_prefix,
):
    """
    Download ECMWF medium-range forecast data.

    Args:
    start_date (datetime): The start date for data retrieval.
    static_path (str): Location of static data.
    mask (xarray.Dataset): Bathymetry mask.
    request_variables (list): List of variables to download.
    ds_variables (list): List of variables in the dataset.
    max_step (int): Maximum time step in hours.
    model (str): Name of the model that produced the data.
    product (str): Forecast product.
    path_prefix (str): Path where the files will be saved.
    """
    grid_mask = np.load(f"{static_path}/sea_mask.npy")[0]

    # Set up HRES client
    client = eo.Client(
        source="ecmwf",
        model=model,
        resol="0p25",
    )

    grib_filename = f"{path_prefix}/{start_date.strftime('%Y%m%d')}.grib"

    # Retrieve data
    client.retrieve(
        date=start_date.strftime("%Y-%m-%d"),
        time=0,
        type=product,
        step=list(range(0, max_step + 1, 6)),
        param=request_variables,
        target=grib_filename,
    )
    print(f"Downloaded {grib_filename}")

    # Open the datasets and drop conflicting height
    datasets = []
    for var in request_variables:
        filter_keys = {"shortName": var}
        ds = xr.open_dataset(
            grib_filename, engine="cfgrib", filter_by_keys=filter_keys
        )  # t, h, w
        ds = ds.drop_vars(["heightAboveGround"], errors="ignore")
        datasets.append(ds)
    merged_ds = xr.merge(datasets)

    # Resample to daily averages
    daily_ds = merged_ds.resample(valid_time="1D").mean()

    # Interpolate onto the sea coordinates
    interp_daily_ds = daily_ds.interp(
        longitude=mask.longitude, latitude=mask.latitude
    )

    # Apply the bathymetry mask to select the exact area
    masked_data = select(interp_daily_ds, mask)

    # Stack as a numpy array
    combined_data = []
    for var in ds_variables:
        data = masked_data[var].values  # t, h, w
        combined_data.append(data)

    combined_data = np.stack(combined_data, axis=-1)  # t, h, w, f
    clean_data = np.nan_to_num(combined_data, nan=0.0)

    # Select only sea grid points
    sea_data = clean_data[:, grid_mask, :]  # t, n_grid, f

    filename = f"{path_prefix}/{start_date.strftime('%Y%m%d')}.npy"
    np.save(filename, sea_data)
    print(f"Saved forecast data to {filename}")


def main():
    """
    Main function to organize the download and processing of oceanographic data.
    """
    parser = argparse.ArgumentParser(
        description="Download oceanographic data."
        )
    parser.add_argument(
        "-b",
        "--base_path",
        type=str,
        default="data/atlantic/",
        help="Output directory",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        default="1987-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        default="2024-05-25",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-d",
        "--data_source",
        type=str,
        choices=["analysis", "reanalysis", "era5"],
        help="Choose between analysis, reanalysis or era5",
    )
    parser.add_argument(
        "--static", action="store_true", help="Download static data"
    )
    parser.add_argument(
        "--forecast", action="store_true", help="Download today's forecast"
    )
    parser.add_argument(
        "-u",
        "--user", 
        type=str,
        help="User name credential for CMEMS"
    )
    parser.add_argument(
        "-psw",
        "--password", 
        type=str,
        help="password credential for CMEMS"
    )
    args = parser.parse_args()

    if args.forecast:
        start_date = datetime.today()
        end_date = start_date + timedelta(days=9)
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    static_path = args.base_path + "static/"
    raw_path = args.base_path + "raw/"
    reanalysis_path = raw_path + "reanalysis"
    analysis_path = raw_path + "analysis"
    era5_path = raw_path + "era5"
    hres_path = raw_path + "hres"
    ens_path = raw_path + "ens"
    aifs_path = raw_path + "aifs"
    forecast_path = raw_path + "forecast"
    bathymetry_mask_path = static_path + "bathy_mask.nc"
    mask = load_mask(bathymetry_mask_path)
    os.makedirs(static_path, exist_ok=True)
    os.makedirs(reanalysis_path, exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(era5_path, exist_ok=True)
    os.makedirs(hres_path, exist_ok=True)
    os.makedirs(ens_path, exist_ok=True)
    os.makedirs(aifs_path, exist_ok=True)
    os.makedirs(forecast_path, exist_ok=True)

    if args.static:
        if "atlantic" in args.base_path.lower():
            # Use the specific function for Atlantic
            download_static_atlantic(static_path, args.user, args.password)
            print("Static files for the Atlantic dataset generated.")
        else:
            # For Mediterranean, it is assumed that the bathymetry mask is available (created or downloaded)
            mask = load_mask(bathymetry_mask_path)
            download_static_mediterranean(static_path, mask)
            print("Static files for the Mediterranean dataset generated.")

    if args.data_source == "reanalysis":
        if "mediterranean" in args.base_path.lower():
            datasets = {
                "med-cmcc-cur-rean-d": ["uo", "vo"],
                "med-cmcc-mld-rean-d": ["mlotst"],
                "med-cmcc-sal-rean-d": ["so"],
                "med-cmcc-ssh-rean-d": ["zos"],
                "med-cmcc-tem-rean-d": ["thetao", "bottomT"],
            }
            version = "202012"

            # start_date = datetime(1987, 1, 1)
            # end_date = datetime(2022, 7, 31)

            download_data(
                start_date,
                end_date,
                datasets,
                version,
                static_path,
                reanalysis_path,
                mask,
                args.user,
                args.password,
            )
        else:
            datasets = {
            "cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE": ["analysed_sst"]
            }
            version = "202411"
            download_data(
                start_date,
                end_date,
                datasets,
                version,
                static_path,
                reanalysis_path,
                mask,
                args.user,
                args.password,
            )

    if args.data_source == "analysis":
        datasets = {
            "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m": ["uo", "vo"],
            "cmems_mod_med_phy-mld_anfc_4.2km_P1D-m": ["mlotst"],
            "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m": ["so"],
            "cmems_mod_med_phy-ssh_anfc_4.2km_P1D-m": ["zos"],
            "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m": ["thetao", "bottomT"],
        }
        version = "202411"

        # start_date = datetime(2021, 11, 1)
        # end_date = datetime(2024, 5, 25)

        download_data(
            start_date,
            end_date,
            datasets,
            version,
            static_path,
            analysis_path,
            mask,
            args.user,
            args.password,
        )

    if args.data_source == "era5":
        request_variables = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind"
        ]
        ds_variables = ["u10", "v10"]
        download_era5(
            start_date,
            end_date,
            request_variables,
            ds_variables,
            static_path,
            era5_path,
            mask,
        )

    if args.forecast:
        datasets = {
            "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m": ["uo", "vo"],
            "cmems_mod_med_phy-mld_anfc_4.2km_P1D-m": ["mlotst"],
            "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m": ["so"],
            "cmems_mod_med_phy-ssh_anfc_4.2km_P1D-m": ["zos"],
            "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m": ["thetao", "bottomT"],
        }
        version = "202311"

        download_forecast(
            start_date,
            end_date,
            datasets,
            version,
            static_path,
            forecast_path,
            mask,
        )

        requests = ["10u", "10v", "2t", "msl"]
        variables = ["u10", "v10", "t2m", "msl"]

        atm_requests = [
            {
                "path": hres_path,
                "requests": requests,
                "variables": variables,
                "model": "ifs",
                "product": "fc",
                "max_step": 240,
            },
            {
                "path": ens_path,
                "requests": requests,
                "variables": variables,
                "model": "ifs",
                "product": "cf",
                "max_step": 360,
            },
            {
                "path": aifs_path,
                "requests": requests,
                "variables": variables,
                "model": "aifs",
                "product": "fc",
                "max_step": 360,
            },
        ]

        for req in atm_requests:
            download_ecmwf_forecast(
                start_date,
                static_path,
                mask,
                req["requests"],
                req["variables"],
                req["max_step"],
                req["model"],
                req["product"],
                req["path"],
            )


if __name__ == "__main__":
    main()
