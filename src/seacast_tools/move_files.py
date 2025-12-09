import mesh_models.mesh_utils as mu
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move graph files to active directory")
    parser.add_argument(
        "--graph_type",
        type=str,
        default="hierarchical",
        help="Type of graph to move (default: hierarchical)"
    )

    parser.add_argument(
        "--graph",
        type=str,
        default="random",
        choices=["random", "uniform", "bathymetry", "fps"],
        help="type of mesh to activate, it also reflects the name of the graph " \
        "folder for soirce_directory (default: random)"
    )
    args = parser.parse_args()
    # Define the source and destination directories
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_directory = os.path.join(root_dir, args.graph_type, "graphs", args.graph)
    target_directory = os.path.join(root_dir, "graphs", args.graph_type)
    
    mu.move_graphfiles_to_active_directory(source_directory, target_directory)
    print(f"Moved graph files from {source_directory} to {target_directory}.")