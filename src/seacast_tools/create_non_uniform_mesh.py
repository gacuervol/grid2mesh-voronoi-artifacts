from mesh_models.Random_mesh import RandomMesh
from mesh_models.Bathy_mesh import BathyMesh
from mesh_models.Uniform_mesh import UniformMesh
from mesh_models.FPS_mesh import FPSMesh
import mesh_models.fps_strategies as fps_strategies
import mesh_models.density_distributions as density_distributions
import mesh_models.mesh_metrics as mm
from argparse import ArgumentParser
import os
import numpy as np

def load_supplementary_masks(supplementary_dir):
    masks = {}
    for file in os.listdir(supplementary_dir):
        if file.endswith(".npy"):
            mask_name = file.split(".")[0]
            masks[mask_name] = np.load(os.path.join(supplementary_dir, file))
            print(f"Loaded mask: {mask_name}")
    return masks

def create_specific_zone_mask(sea_surface_2d: np.ndarray,
                              x_coords,
                              y_coords) -> np.ndarray:
    """
    Creates a 2D mask with the shape defined by x_coords and y_coords,
    with True where there is sea and inside the specified area.

    Parameters
    ----------
    sea_surface_2d : np.ndarray(bool), shape (M, N)
        Sea mask over the entire grid (True where there is sea).

    x_coords : array-like of ints, shape (n_x,)
        Column indices of the specified zone.

    y_coords : array-like of ints, shape (n_y,)
        Row indices of the specified zone.

    Returns
    -------
    np.ndarray(bool), shape (n_y, n_x)
        2D mask of the specified zone (True where there is sea and within the zone).
    """

    # Ensure x_coords and y_coords are numpy arrays
    x_coords = np.asarray(x_coords, dtype=int).ravel()
    y_coords = np.asarray(y_coords, dtype=int).ravel()

    # Check if coordinates are within the bounds of the sea_surface_2d
    max_y, max_x = sea_surface_2d.shape
    if x_coords.min() < 0 or x_coords.max() >= max_x:
        raise IndexError(f"x_coords fuera de rango: deben estar entre 0 y {max_x - 1}")
    if y_coords.min() < 0 or y_coords.max() >= max_y:
        raise IndexError(f"y_coords fuera de rango: deben estar entre 0 y {max_y - 1}")

    # Extract the specific zone mask
    mask_zone = sea_surface_2d[np.ix_(y_coords, x_coords)]

    return mask_zone

def create_specific_zone_mask_grid(sea_surface_2d: np.ndarray,
                              x_coords,
                              y_coords) -> np.ndarray:

    mask = np.zeros(sea_surface_2d.shape, dtype=bool)
    mask[y_coords[:, None], x_coords] = True
    # apply the sea_mask
    mask &= sea_surface_2d
    return mask


def create_supplementary_masks(supplementary_mask_dir, sea_mask):
    white_cape_x = np.arange(0, 100)
    white_cape_y = np.arange(0, 55)

    upwelling_zone = mm.get_african_upwelling_zone(sea_mask, n_pixels=10)
    ocean_non_upw = np.logical_and(sea_mask, ~upwelling_zone)
    white_cape = create_specific_zone_mask_grid(sea_mask, 
                                            x_coords=white_cape_x,
                                            y_coords=white_cape_y)
    ghir_cape_x = np.arange(170, 240)
    ghir_cape_y = np.arange(170, 250)
    ghir_cape = create_specific_zone_mask_grid(sea_mask,
                                            x_coords=ghir_cape_x,
                                            y_coords=ghir_cape_y)
    juby_cape_x = np.arange(105, 240)
    juby_cape_y = np.arange(105, 180)
    juby_cape = create_specific_zone_mask_grid(sea_mask,
                                            x_coords=juby_cape_x,
                                            y_coords=juby_cape_y)
    np.save(os.path.join(supplementary_mask_dir, "white_cape.npy"), white_cape)
    np.save(os.path.join(supplementary_mask_dir, "ghir_cape.npy"), ghir_cape)
    np.save(os.path.join(supplementary_mask_dir, "juby_cape.npy"), juby_cape)
    np.save(os.path.join(supplementary_mask_dir, "upwelling_zone.npy"), upwelling_zone)
    np.save(os.path.join(supplementary_mask_dir, "non_upwelling_ocean.npy"), ocean_non_upw)
    supplementary_masks = {
        "white_cape": white_cape,
        "ghir_cape": ghir_cape,
        "juby_cape": juby_cape,
        "upwelling_zone": upwelling_zone,
        "non_upwelling_ocean": ocean_non_upw
    }
    return supplementary_masks
def calculate_mesh_levels(args, xy, nx):
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    mesh_levels = nlev -1
    if args.levels is not None:
        mesh_levels = min(mesh_levels, args.levels)
    return mesh_levels

if __name__ == "__main__":
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="atlantic",
        help="Dataset to load grid point coordinates from "
        "(default: atlantic)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical",
        help="Name to save graph as (default: hierarchical)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If graphs should be plotted during generation "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=1,
        help="Generate hierarchical mesh graph (default: 1, yes)",
    )
    parser.add_argument(
        "--mesh_type",
        type=str,
        choices=["random", "bathymetry", "uniform", "fps"],
        default="uniform",
        help="Mesh type to use (default: uniform)",
    )
    parser.add_argument(
        "--probability_distribution",
        type=str,
        choices=["mixed_sigmoid", "base"],
        default="mixed_sigmoid",
        help="probability distribution to use, currently implemented" \
        "for bathymetry mesh only (default: mixed_sigmoid)",
    )
    parser.add_argument(
        "--crossing_edges",
        type=int,
        default=0,
        help="If crossing edges should be added to the mesh. This argument is currently for the uniform mesh only where:" \
        "- 0: no crossing edges (default) where we add edges diagonally with Delaunny" \
        "- 1: crossing edges are added to the mesh in an x shape (default: 0)",
    )
    parser.add_argument(
        "--uniform_resolution_list",
        type =lambda s: [int(item) for item in s.split(',')],
        default=[81, 27, 9],
        help="List of resolutions for the uniform mesh levels (default: [81, 27, 9])" \
        "The list is a list of integers with the number of nodes in each level." \
        "The length of the list should be equal to the number of levels." \
        ,
        )
    parser.add_argument(
        "--n_connections",
        type=int,
        default=1,
        help="Number of connections for the g2m and m2g graphs (default: 1)",
    )
    parser.add_argument(
        "--k_neighboors",
        type=int,
        default=1,
        help="Number of neighbors for the up and down edges (default: 1)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["fps", "fps_weighted"],
        default="fps",
        help="Sampling strategy to use for FPS mesh (default: fps)",
    )
    parser.add_argument(
        "--nodes_amount",
        type =lambda s: [int(item) for item in s.split(',')],
        default=[3568, 394, 45], #values taken from the original 81, 27, 9 mesh
        help="Number of nodes to use for each level in mesh of the non uniform mesh if fixed amount is wanted" \
        "The list is a list of integers with the number of nodes in each level." \
        "the compatible meshes are all except the uniform mesh."
    )

    args = parser.parse_args()

    print("uniform_resolution_list", args.uniform_resolution_list)
    #root_dir should change depending on the position of the script
    #if this is moved to same level as create_mesh.py, root_dir is no longer necessary
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_dir_path = os.path.join(root_dir, "data", args.dataset, "static")
    original_graphs_dir_path = os.path.join(root_dir, "graphs", args.graph)
    xy = np.load(os.path.join(static_dir_path, "nwp_xy.npy"))
    sea_mask = np.load(os.path.join(static_dir_path, "sea_mask.npy"))

    #load supplementary masks
    supplementary_dir = os.path.join("mesh_models", "supplementary_masks")
    if os.path.exists(supplementary_dir):
        supplementary_masks = load_supplementary_masks(supplementary_dir)
    else:
        os.makedirs(supplementary_dir, exist_ok=True)
        supplementary_masks = create_supplementary_masks(supplementary_dir,sea_mask[0])
    land_mask = ~sea_mask[0]
    
    base_dir_path_to_save= os.path.join(root_dir, args.graph)

    #for fixed experiments with 3 levels
    fixed_total_nodes = True
    #total_nodes_for_each_level = {81: 3568, 27: 394, 9: 45} #81,27,9
    #total_nodes_for_each_level = {81: 2158, 27: 243, 9: 27} #63, 21, 7
    #total_nodes_for_each_level = {81: 1103, 27: 122, 9: 14} #45, 15, 5
    #total_nodes_for_each_level = {81: 488, 27: 122, 9: 14} # 30, 15, 5
    #total_nodes_for_each_level = {81: 394, 27: 45, 9: 5} #27,9,3
    #total_nodes_for_each_level = {81: 80, 27: 20, 9: 5} #12, 6, 3
    #total_nodes_for_each_level = {81:  14, 27: 9, 9: 5} #5, 4, 3
    #use args.nodes_amount for the fixed amount of nodes
    if args.mesh_type != "uniform":
        """
        keys are currently hardcoded as divisors of 243, where first one is 81,
        this should be changed to be more flexible in the future.
        """
        base_number = 81
        total_nodes_for_each_level = {}
        for i, nodes in enumerate(args.nodes_amount):
            if i == 0:
                total_nodes_for_each_level[base_number] = nodes
            else:
                base_number = int(base_number / 3)
                total_nodes_for_each_level[base_number] = nodes

    bathymetry = np.load(os.path.join(static_dir_path, "sea_depth.npy"))
        
    sea_depth_norm = (bathymetry - bathymetry.min()) / (bathymetry.max() - bathymetry.min())
    
    if args.mesh_type == "random":
        nx = 3
        mesh_levels = calculate_mesh_levels(args, xy, nx)
        mesh = RandomMesh(xy, nx, land_mask, args.mesh_type, 
                          base_dir_path_to_save, seed = 42, 
                          fixed_total_nodes=fixed_total_nodes,
                          total_nodes_for_each_level=total_nodes_for_each_level, 
                          supplementary_masks=supplementary_masks)

        graph_list = mesh.create_mesh_levels(mesh_levels, plot = bool(args.plot))
        graph_list, first_index_level = mesh.prepare_hierarchical_graphs(graph_list)
        mesh.create_hierarchical_mesh_up_mesh_down_edges(graph_list, mesh_levels, 
                                                        first_index_level, plot = bool(args.plot),
                                                        k_neighboors = args.k_neighboors)
        m2m_graphs = mesh.create_hierarchical_m2m_graph(graph_list, first_index_level=first_index_level)
        mesh.create_mesh_features(m2m_graphs)
        G_g2m = mesh.create_g2m_graph(graph_list, plot = bool(args.plot), required_connections=args.n_connections)
        mesh.create_m2g_graph(G_g2m, plot= bool(args.plot))

    elif args.mesh_type == "bathymetry":
        nx = 3
        mesh_levels = calculate_mesh_levels(args, xy, nx)
        mesh = BathyMesh(xy, nx, land_mask, args.mesh_type, 
                          base_dir_path_to_save, bathymetry=sea_depth_norm,seed = 42, 
                          fixed_total_nodes=fixed_total_nodes,
                          total_nodes_for_each_level=total_nodes_for_each_level, 
                          distribution_type=args.probability_distribution, 
                          supplementary_masks=supplementary_masks)

        graph_list = mesh.create_mesh_levels(mesh_levels, plot = bool(args.plot))
        graph_list, first_index_level = mesh.prepare_hierarchical_graphs(graph_list)
        mesh.create_hierarchical_mesh_up_mesh_down_edges(graph_list, mesh_levels, 
                                                        first_index_level, plot = bool(args.plot),
                                                        k_neighboors = args.k_neighboors)
        m2m_graphs = mesh.create_hierarchical_m2m_graph(graph_list, first_index_level=first_index_level)
        mesh.create_mesh_features(m2m_graphs)
        G_g2m = mesh.create_g2m_graph(graph_list, plot = bool(args.plot), required_connections=args.n_connections)
        mesh.create_m2g_graph(G_g2m, plot= bool(args.plot))

    elif args.mesh_type == "uniform":
        nx = 3
        mesh_levels = calculate_mesh_levels(args, xy, nx)
        mesh = UniformMesh(xy, nx, land_mask, args.mesh_type,
                          base_dir_path_to_save, supplementary_masks=supplementary_masks, 
                          crossing_edeges=bool(args.crossing_edges))
        graph_list = mesh.create_mesh_levels(mesh_levels, plot = bool(args.plot), resolutions_list=args.uniform_resolution_list)
        graph_list, first_index_level = mesh.prepare_hierarchical_graphs(graph_list)
        mesh.create_hierarchical_mesh_up_mesh_down_edges(graph_list, mesh_levels, 
                                                        first_index_level, plot = bool(args.plot),
                                                        k_neighboors = args.k_neighboors)
        m2m_graphs = mesh.create_hierarchical_m2m_graph(graph_list, first_index_level=first_index_level)
        mesh.create_mesh_features(m2m_graphs)
        G_g2m = mesh.create_g2m_graph(graph_list, plot = bool(args.plot), 
                                      required_connections=args.n_connections)
        mesh.create_m2g_graph(G_g2m, plot= bool(args.plot))
    
    elif args.mesh_type == "fps":
        nx = 3
        mesh_levels = calculate_mesh_levels(args, xy, nx)
        mesh = FPSMesh(xy, nx, land_mask, args.mesh_type, 
                          base_dir_path_to_save, bathymetry=bathymetry,seed = 42, 
                          sampler = args.sampler,
                          distribution_type=args.probability_distribution,
                          fixed_total_nodes=fixed_total_nodes,
                          total_nodes_for_each_level=total_nodes_for_each_level,  
                          supplementary_masks=supplementary_masks
                        )
        graph_list = mesh.create_mesh_levels(mesh_levels, plot = bool(args.plot))
        graph_list, first_index_level = mesh.prepare_hierarchical_graphs(graph_list)
        mesh.create_hierarchical_mesh_up_mesh_down_edges(graph_list, mesh_levels, 
                                                        first_index_level, plot = bool(args.plot),
                                                        k_neighboors = args.k_neighboors)
        m2m_graphs = mesh.create_hierarchical_m2m_graph(graph_list, first_index_level=first_index_level)
        mesh.create_mesh_features(m2m_graphs)
        G_g2m = mesh.create_g2m_graph(graph_list, plot = bool(args.plot), 
                                      required_connections=args.n_connections)
        mesh.create_m2g_graph(G_g2m, plot= bool(args.plot))