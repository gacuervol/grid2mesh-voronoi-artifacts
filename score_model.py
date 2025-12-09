# Standard library
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Third-party
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX import aggregation

# First-party
from src.seacast_tools.np_loaders import TargetsFromNumpy, PredictionsFromNumpy

def main():
    """Main function to run the inference model and compute metrics.
    """
    # Define argument parser
    parser = ArgumentParser(description="Run inference model and compute metrics.")
    # Add arguments for directories e.i. ${HOME}/output/crossing_edges_5_3/predictions
    parser.add_argument("--pred_dir", type=str, help="Path to predictions directory.")
    # Add arguments for test directory e.i. ${HOME}/data/atlantic/samples/test
    parser.add_argument("--test_dir", type=str, help="Path to test directory.")
    # Add arguments for variables e.i. ["sst_temperature"]
    parser.add_argument("--variables", type=str, nargs='+', help='List of variables to evaluate e.i. "sst_temperature sea_ice_concentration".')

    # Parse arguments
    args = parser.parse_args()

    # Deterministic metrics for evaluation
    metrics = {
        'rmse': deterministic.RMSE(),
        'mae': deterministic.MAE(),
        'bias': deterministic.Bias(),
        }
    # Load predictions and targets
    # Load masks and coordinates
    extra_variables = ["coordinates", "mask"]

    extra_files = [
        f"./data/atlantic/static/coordinates.npy", 
        f"./data/atlantic/static/sea_mask.npy"
        ]
    # Define predictions loader
    pred_loader = PredictionsFromNumpy(
        path=args.pred_dir,
        variables=args.variables,
        extra_variables=extra_variables,
        extra_files=extra_files
        )
    # Load predictions dataset
    predictions_dataset = pred_loader.load_chunk()

    # Define targets loader
    targets_loader = TargetsFromNumpy(
        path=args.test_dir,
        variables=args.variables,
        extra_variables=extra_variables,
        extra_files=extra_files
        )
    # Load targets dataset
    targets_dataset = targets_loader.load_chunk()

    # Compute statistics for all metrics
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, predictions_dataset, targets_dataset
        )
    # Save each score and variable statistics
    stats_dir = os.path.join(os.path.dirname(args.pred_dir), "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    for score_name, variable_dic in statistics.items():
        for variable_name, variable_stats in variable_dic.items():
            out_file = os.path.join(stats_dir, f"{score_name}_{variable_name}.nc")
            variable_stats.to_netcdf(out_file, engine="netcdf4")
            print(f":::: Saved statistics for {score_name} - {variable_name} to {out_file}")
if __name__ == "__main__":
    main()