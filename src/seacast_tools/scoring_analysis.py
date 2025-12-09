# This module provides classes and functions to load, aggregate, and plot
# statistical scores from NetCDF files in a specified directory.
# Standard library imports

# Standard library imports
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import Dict, Union, Tuple, List, Optional, Callable
from collections import defaultdict
from functools import partial

# Third-party imports
import xarray as xr
from weatherbenchX import aggregation

# Loader for predictions and targets
class StatsLoader:
    """ A class to load and manage statistics from a directory containing NetCDF files.
    Attributes:
        name (str): Name of the experiment, derived from the path.
        stats_dir (os.PathLike): Path to the statistics directory.
    """
    def __init__(self, exp_path: os.PathLike):
        self.name: str = os.path.basename(exp_path)
        self.stats_dir: os.PathLike = self._get_stats_dir(exp_path)

    def __repr__(self) -> str:
        return f"ExpStats(name={self.name}, stats_dir={self.stats_dir})"

    def __str__(self) -> str:
        return f"Experiment: {self.name}, Statistics Directory: {self.stats_dir}"
    
    @staticmethod
    def _get_stats_dir(exp_path: os.PathLike) -> os.PathLike:
        """
        Returns the path to the statistics directory.

        Returns:
            os.PathLike: Path to the statistics directory.
        """
        return os.path.expanduser(os.path.join(exp_path, "statistics"))

    @property
    def score_names(self) -> set:
        """
        Returns a set of unique score names from the statistics directory.

        Returns:
            set: Unique score names.
        """
        return self._get_scores_and_variable_names(self.stats_dir)[0]

    @property
    def variable_names(self) -> set:
        """
        Returns a set of unique variable names from the statistics directory.

        Returns:
            set: Unique variable names.
        """
        return self._get_scores_and_variable_names(self.stats_dir)[1]
    
    @staticmethod
    def _get_scores_and_variable_names(stats_dir: os.PathLike) -> Tuple[set, set]:
        """
        Get unique score names and variable names from the statistics directory.

        Args:
            stats_dir (str): Path to the directory containing statistics files.

        Returns:
            tuple: (set of score names, set of variable names)
        """
        scores = set()
        variables = set()
        expanded_path = os.path.expanduser(stats_dir)
        for filename in os.listdir(expanded_path):
            if "_" in filename:
                score, var = filename.split('_', 1)
                var = var.replace('.nc', '')
                scores.add(score)
                variables.add(var)
        
        return scores, variables

    def load_statistics(self) -> Dict[str, Dict[str, xr.Dataset]]:
        """
        Loads score NetCDF files from a directory and organizes the datasets
        into a nested dictionary: {score_name: {variable_name: Dataset}}.

        Parameters:
            stats_dir (str): Path to the directory containing the .nc score files.

        Returns:
            dict[str, dict[str, xarray.Dataset]]
        """
        statistics = defaultdict(dict)

        for filename in os.listdir(self.stats_dir):
            if filename.endswith(".nc"):
                name = filename.replace('.nc', '')  # Remove ".nc"
                # split into score and variable names
                score_name, variable_name = name.split('_', 1) 
                # Store current path
                filepath = os.path.join(self.stats_dir, filename)
                # Add the dataarray to the nested dictionary
                statistics[score_name][variable_name] = xr.load_dataarray(filepath)

        return dict(statistics)

# Aggregation session for statistics
class StatsAggSession:
    """ A class to manage the aggregation of statistics for a specific experiment.
    Attributes:
        exp_stats (StatsLoader): An instance of StatsLoader for the experiment.
        exp_name (str): Name of the experiment.
        score_names (set): Set of score names in the experiment.
        variable_names (set): Set of variable names in the experiment.
    """
    def __init__(self, exp_stats: StatsLoader):
        self.exp_stats = exp_stats

    def __enter__(self):
        self.exp_name = self.exp_stats.name
        self.score_names = self.exp_stats.score_names
        self.variable_names = self.exp_stats.variable_names
        self._load_stats_fn = self.exp_stats.load_statistics
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__dict__.clear() # Clear all attributes to free memory
        import gc; gc.collect() # Force garbage collection to free memory

    def __repr__(self) -> str:
        return f"StatsAggSession(exp_name={self.exp_name}, score_names={self.score_names}, variable_names={self.variable_names})"   

    def __str__(self) -> str:
        return f"Stats. Aggregation Session for Experiment: {self.exp_name}, Scores: {self.score_names}, Variables: {self.variable_names}"
    
    def agg_scores(
        self, 
        reduce_dims: List[str],
        ) -> Dict[str, Dict[str, xr.Dataset]]:
        """
        Aggregates the experiment statistics by reducing the specified dimensions.

        Args:
            reduce_dims (List[str]): List of dimension names to reduce (e.g., ["init_time", "latitude"]).

        Returns:
            Dict[str, Dict[str, xr.Dataset]]: Aggregated datasets grouped by score and variable.
        """
        aggregator = aggregation.Aggregator(reduce_dims=reduce_dims, skipna=True)
        aggregation_state = aggregator.aggregate_statistics(self._load_stats_fn())
        return aggregation_state.mean_statistics()

# Plotting functions for statistics
def plot_lead_time_scores(
    exp_name: str,
    score_names: Union[set, List], 
    variable_names: Union[set, List], 
    lead_time_scores: Dict[str, Dict[str, xr.Dataset]],
    axs: plt.Axes, 
    ) -> plt.Axes:
    """
    Plots the lead time scores for a given score and variable.

    Args:
        score_name (str): The name of the score to plot.
        variable_name (str): The name of the variable to plot.
    """

    # Number of axes must be a higher than the number of scores
    axs_f = axs.flatten() if hasattr(axs, 'flatten') else axs
    if len(score_names) > len(axs_f):
        raise ValueError(
            f"Number of scores ({len(score_names)}) exceeds number of axes ({len(axs_f)})."
            )
    # Plot each score and variable in the grid
    for ax, score in zip(axs_f, score_names):
        for variable in variable_names:
            data = lead_time_scores[score][variable]
            lead_days = data.lead_time.data.astype('timedelta64[D]')
            ax.plot(lead_days, data.data, marker='o', linestyle='-', label=exp_name)
            ax.set_xlabel("Lead Time (days)")
            ax.set_ylabel(f"{score} - {variable} (K)")
            ax.legend()

    return axs

def plot_map_scores(
    exp_name: str,
    score_name: str, 
    variable_names: Union[set, List], 
    map_scores: Dict[str, Dict[str, xr.Dataset]],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    n_cols: int = 5,
    cmap: Union[str, Colormap] = 'bone_r',
    ) -> plt.Axes:
    """
    Plots the map scores for a given score and variable.

    Args:
        score_name (str): The name of the score to plot.
        variable_name (str): The name of the variable to plot.
    """

    # Plot each score and variable in the grid
    for variable in variable_names:
        map_scores[score_name][variable].plot(
        x="longitude", 
        y="latitude", 
        col="lead_time", 
        cmap='coolwarm' if score_name == 'Error' else cmap,
        col_wrap=n_cols,
        aspect=1, 
        size=4,
        rasterized=True,
        vmin=vmin,  # Adjust as needed
        vmax=vmax,  # Adjust as needed
        )
        plt.suptitle(f"{exp_name} {score_name} - {variable}", fontsize=16, y=1.02)

def filter_statistics(
    aggregated_stats: Dict[str, Dict[str, xr.Dataset]],
    dim_name: str,
    filter_idx: Optional[List[int]],
    ) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Filters and optionally slices the aggregated statistics by lead time indices.
    """
    # Check if the dimension exists in the datasets.
    check_fn = partial(_check_dimension, dim_name=dim_name)
    _map_structure(aggregated_stats, check_fn)
    # Filter the datasets based on the specified dimension and indices.
    filter_fn = partial(_filter, dim_name=dim_name, filter_idx=filter_idx)
    aggregated_stats = _map_structure(aggregated_stats, filter_fn)

    return aggregated_stats

def get_min_max(
    aggregated_stats: Dict[str, Dict[str, xr.Dataset]],
    ) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Get the maximum value of each dataset in the aggregated statistics. 
    """
    return (
        _map_structure(aggregated_stats, _get_min),
        _map_structure(aggregated_stats, _get_max)
        )

    return min_stats, max_stats

def _map_structure(
    structure: Dict[str, Dict[str, xr.Dataset]],
    func: Callable[[xr.Dataset], xr.Dataset]
    ) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Applies a function to each dataset in the nested dictionary structure.

    Args:
        structure (Dict[str, Dict[str, xr.Dataset]]): The nested dictionary structure.
        func (callable): The function to apply to each dataset.

    Returns:
        Dict[str, Dict[str, xr.Dataset]]: The modified structure with the function applied.
    """
    new_structure = defaultdict(dict)
    # Iterate through the structure and apply the function to each dataset
    for score, variables in structure.items():
        for variable, dataset in variables.items():
            new_structure[score][variable] = func(dataset)

    return new_structure

def _check_dimension(
    dataset: xr.Dataset,
    dim_name: str,
    ) -> xr.Dataset:
    """
    Checks if the specified dimension exists in the dataset for the given score and variable.
    """ 
    if dim_name not in dataset.dims:
        raise ValueError(
            f"Dimension '{dim_name}' not found in the dataset."
            )
    return dataset

def _filter(
    dataset: xr.Dataset,
    dim_name: str,
    filter_idx: List[int],
    ) -> xr.Dataset:
    """
    Filters and slices the dataset based on the specified dimension and indices.
    """
    return dataset.isel({dim_name: filter_idx})

def _get_min(dataset: xr.Dataset) -> float:
    """
    Returns the minimum and maximum values of the dataset.
    """
    return float(dataset.min().data)

def _get_max(dataset: xr.Dataset) -> float:
    """
    Returns the maximum value of the dataset.
    """
    return float(dataset.max().data)