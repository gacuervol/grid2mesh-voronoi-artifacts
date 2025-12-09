# Standard library
import os
import numpy as np
from datetime import datetime
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Union, List

# Third-party
import xarray as xr
from weatherbenchX import interpolations
import weatherbenchX.data_loaders.base as base

# First-party
from src.seacast_tools import base_loader_mod


def _extract_date_from_filename(filename: str) -> str:
    date_str = filename.rsplit('_', 1)[-1].replace('.npy', '')
    date_dt = datetime.strptime(date_str, "%Y%m%d")
    return np.datetime64(date_dt.strftime("%Y-%m-%dT00:00:00")) - np.timedelta64(1, 'D')

def _load_numpy_file(source: Union[str, List[str]]) -> List:
    """Load numpy files from a directory or a list of files and return the data as a list."""

    list_files = []
    date_filenames = []

    if isinstance(source, str) and os.path.isdir(source):
        filenames = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.npy')]

        for filename in filenames:
            if 'rea_data' in filename:
                date_filenames.append(_extract_date_from_filename(filename))

    elif isinstance(source,list):
        filenames = [f for f in source if f.endswith('.npy')]
    else:
        raise ValueError("The argument must be a directory or a list of files.")
    
    for file_path in filenames:
        if "rea_data" in file_path or type(source) == list:
            data = np.load(file_path)
            list_files.append(data)

    return list_files, date_filenames

def _process_coords(coords: np.ndarray):

    lat_grid = coords[0]
    lon_grid = coords[1]

    lat_grid = lat_grid[:, 0]
    lon_grid = lon_grid[0, :]

    return lat_grid, lon_grid

def _apply_mask_to_entry(data_entry: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a mask to a single data entry, filling unmasked areas with NaN."""
    data_entry = data_entry.squeeze()
    data_grid = np.full(mask.shape, np.nan, dtype=np.float32)
    data_grid[mask == 1] = data_entry
    return data_grid


def _apply_mask(data: List, mask: np.ndarray) -> List:
    """Apply a mask to the data, filling unmasked areas with NaN."""
    masked_data = []

    mask = mask.astype(int)

    for data_entries in data:

        data_entries_proccessed = []

        for data_entry in data_entries:
            data_entry = _apply_mask_to_entry(data_entry, mask)
            data_entries_proccessed.append(data_entry)
        
        data_entries_proccessed = np.stack(data_entries_proccessed, axis=0)
        data_entries_proccessed = data_entries_proccessed.squeeze()

        if data_entries_proccessed.shape[0] > 15:
            data_entries_proccessed = data_entries_proccessed[2:, :, :]

        masked_data.append(data_entries_proccessed)

    return masked_data



def _build_XarrayDataset(data: List, variables: Iterable[str], coords: tuple, init_times: Iterable[str], mask = np.ndarray, valid_time: bool = False) -> xr.Dataset:

    """Build an Xarray Dataset from the data and variables."""
    
    variables = list(variables)

    init_times= np.array(init_times, dtype='datetime64[ns]')

    lead_times = np.array([np.timedelta64(24 * (i+1), 'h') for i in range(15)], dtype='timedelta64[ns]')

    latitude, longitude = coords

    mask = np.tile(mask, (len(init_times), len(lead_times), 1, 1))

    dataset_vars = {
        var: (["init_time", "lead_time", "latitude", "longitude"], data_entry)
        for var, data_entry in zip(variables, data)
    }

    coords = {
        "latitude": latitude,
        "longitude": longitude,
        "init_time": init_times,
        "lead_time": lead_times,
        "mask": (["init_time", "lead_time", "latitude", "longitude"], mask),
    }

    if valid_time:
        init_times_array = np.array(init_times)[:, np.newaxis]
        lead_times_array = np.array(lead_times)[np.newaxis, :]

        valid_times = init_times_array + lead_times_array 
        coords["valid_time"] = (("init_time", "lead_time"), valid_times)

    dataset = xr.Dataset(
        data_vars=dataset_vars,
        coords=coords,
    )

    return dataset


class NpLoaders(base.DataLoader):
    """Base class for Numpy data loaders"""

    def __init__(
        self, 
        path: str = None,
        variables: Iterable[str]= None,
        extra_files: Optional[Iterable[str]] = None,
        extra_variables: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        
        """Init.

        Args:
        path: (Optional) Path to folder storing numpy files to open. It needs to contain only files with .np extension,
        variables: Give name of the variables stored in the numpy file.
        extra_files: (Optional) List of extra files to load. These files are not used to create the dataset, but they are used to create the mask and the coordinates. The expected files and sorting are:
            - coordinates.npy
            - mask.npy
        extra_variables: (Optional) List of extra variables to load. These variables are not used to create the dataset, but they are used to create the mask and the coordinates. The expected variables and sorting are:
            - coordinates
            - mask
        """
        
        self.path = path
        self.variables = variables
        self.extra_files = extra_files
        self.extra_variables = extra_variables

        self._validate_path()
        self._load_auxiliary_data()
        self._load_and_process_data()

        base_loader_mod.overwrite_load_chunk()


    def _validate_path(self):
        """Validate the path and check if it contains only .npy files."""
        if not os.path.isdir(self.path):
            raise ValueError("The path must be a directory.")
        
        files = os.listdir(self.path)

        only_files = [f for f in files if os.path.isfile(os.path.join(self.path, f))]

        if not all(f.endswith('.npy') for f in only_files):
            raise ValueError("The path must contain only .npy files.")
    
    def _load_auxiliary_data(self):
        """Load auxiliary data from the extra files."""
                
        if (self.extra_variables is not None and self.extra_files is not None) or len(self.extra_variables) > 2:
            if len(self.extra_files) != len(self.extra_variables):
                raise ValueError("The number of extra files must be equal to the number of extra variables. Futhermore, it should be lower than 3 (coordinates, mask).")
            
            _extra_data, _ = _load_numpy_file(self.extra_files)
                
            extra_dict = dict(zip(self.extra_variables, _extra_data))

            try:
                self.latitude, self.longitude = _process_coords(extra_dict["coordinates"])
                self.mask = extra_dict["mask"]
            except KeyError as e:
                raise ValueError(f"Missing required auxiliary variable: {e}")
                
        
    def _load_and_process_data(self):

        _data_entries, self._init_dates = _load_numpy_file(self.path)

        self.all_data = [_apply_mask(_data_entries, self.mask)]


    def _load_chunk_from_source(
        self
    ) -> xr.Dataset:
        xr_dataset = _build_XarrayDataset(self.all_data, self.variables, (self.latitude, self.longitude), self._init_dates, self.mask)

        return xr_dataset
    

class TargetsFromNumpy(NpLoaders):
    """DataLoader for target data in numpy files."""
    
    def __init__(self, path: str, variables: Iterable[str], extra_files: Optional[Iterable[str]] = None, extra_variables: Optional[Iterable[str]] = None):
        super().__init__(path, variables, extra_files, extra_variables)

    def _load_chunk_from_source(self) -> xr.Dataset:
        xr_dataset = _build_XarrayDataset(self.all_data, self.variables, (self.latitude, self.longitude), self._init_dates, self.mask, valid_time=True)
        
        return xr_dataset

class PredictionsFromNumpy(NpLoaders):
    """Dataloader for prediction data in numpy files."""
    
    def __init__(self, path: str, variables: Iterable[str], extra_files: Optional[Iterable[str]] = None, extra_variables: Optional[Iterable[str]] = None):
        super().__init__(path, variables, extra_files, extra_variables)

    def _load_chunk_from_source(self) -> xr.Dataset:
        xr_dataset = _build_XarrayDataset(self.all_data, self.variables, (self.latitude, self.longitude), self._init_dates, self.mask)
        
        return xr_dataset
        
        


        
        
