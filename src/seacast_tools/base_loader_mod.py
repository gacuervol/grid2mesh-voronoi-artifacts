from weatherbenchX.data_loaders.base import DataLoader
from typing import Collection, Hashable, Mapping, Optional, Union
import numpy as np
from weatherbenchX import interpolations
from weatherbenchX import xarray_tree
import xarray as xr


def load_chunk(
      self,
      init_times: Optional[np.ndarray] = None,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
      reference: Optional[Mapping[Hashable, xr.DataArray]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:

    if init_times is None and lead_times is None:
      chunk = self._load_chunk_from_source()
    
    else:
      chunk = self._load_chunk_from_source(init_times, lead_times) 

    if self._interpolation is not None:
      # TODO(srasp): Potentially implement consistency check between lead_times
      # and lead_time coordinate on reference.
      chunk = self._interpolation.interpolate(chunk, reference)

    # Compute after interpolation avoids loading unnecessary data.
    if self._compute:
      chunk = xarray_tree.map_structure(lambda x: x.compute(), chunk)

    if self._add_nan_mask:
      chunk = add_nan_mask_to_data(chunk)

    return chunk

def overwrite_load_chunk():
    """Overwrite the load_chunk method in BaseLoader."""
    DataLoader.load_chunk = load_chunk