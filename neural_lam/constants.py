# Third-party
import cartopy
import numpy as np

WANDB_PROJECT = "seacast"

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 3, 4])
TEST_STEP_LOG_ERRORS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)

# Sample lengths
SAMPLE_LEN = {
    "train": 6,
    "val": 6,
    "test": 17,
}

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = []

# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {}

# Variable names
PARAM_NAMES = [
    "Analysed sea surface temperature",
    #"estimated error standard deviation of analysed_sst"
]

PARAM_NAMES_SHORT = [
    "sea_surface_foundation_temperature"
    #"analysis_error"
]

PARAM_UNITS = [
    "K"
    #"K"
]

PARAM_COLORMAPS = [
    "RdBu_r",
    #"viridis"
]

LEVELS = [
    False,
    #False
]

# Projection and grid
GRID_SHAPE = (300, 300)  # (y, x)
GRID_LIMITS = [-20.97, 19.55, -5.975, 34.525]
#PROJECTION = cartopy.crs.PlateCarree()

# Data dimensions
GRID_FORCING_DIM = 4 * 3 # WARNING
GRID_STATE_DIM = 1 # WARNING

DEPTHS = [
    0.2
] #this constant should not be used cause our data is single-level


# New lists
EXP_PARAM_NAMES_SHORT = []
EXP_PARAM_UNITS = []
EXP_PARAM_COLORMAPS = []

for name, unit, colormap, levels_applies in zip(
    PARAM_NAMES_SHORT, PARAM_UNITS, PARAM_COLORMAPS, LEVELS
):
    if levels_applies:
        for depth in DEPTHS:
            depth_int = round(depth)
            EXP_PARAM_NAMES_SHORT.append(f"{name}_{depth_int}")
            EXP_PARAM_UNITS.append(unit)
            EXP_PARAM_COLORMAPS.append(colormap)
    else:
        EXP_PARAM_NAMES_SHORT.append(name)
        EXP_PARAM_UNITS.append(unit)
        EXP_PARAM_COLORMAPS.append(colormap)
