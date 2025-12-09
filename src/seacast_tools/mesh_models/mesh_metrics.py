"""
Supplementary functions for mesh creations where we can compute mesh metrics
and add different ploting methods to visualize the mesh.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx
import torch_geometric as pyg
from collections import Counter
from scipy.ndimage import distance_transform_edt, label


def get_african_upwelling_zone(sea_surface: np.ndarray,
                               n_pixels: int = 10) -> np.ndarray:
    """
    Gets the upwelling zone of width n_pixels *only* along
    the African coast, excluding islands.

    Parameters
    ----------
    sea_surface : np.ndarray, bool
    Mask where True = sea, False = land.
    n_pixels : int
    Width of the coastal zone in pixels (default 10).

    Returns
    -------
    np.ndarray, bool
    Mask of the upwelling zone: True = sea pixels within
    n_pixels of the African coast.
    """
    sea = sea_surface.astype(bool)
    land = ~sea

    labeled_land, n_components = label(land)

    sizes = np.bincount(labeled_land.ravel())
    sizes[0] = 0
    main_label = sizes.argmax()

    africa_mask = (labeled_land == main_label)

    dist_mask = ~africa_mask

    dist = distance_transform_edt(dist_mask)

    upwelling_zone = sea & (dist <= n_pixels)

    return upwelling_zone


def plot_graph(graph,sea_surface=None,title=None, graph_type=None, save_dir="figures", xlim=None, ylim=None 
):
    """
    Function to plot a graph using matplotlib using a PyG graph object.
    :param graph: PyG graph object to be plotted.
    :param title: Title of the plot.
    :param graph_type: Type of graph (e.g., 'hierarchical', 'combined').
    :param save_dir: Directory to save the plot.
    :param xlim: Tuple (xmin, xmax) to set x-axis limits. If None, uses automatic limits.
    :param ylim: Tuple (ymin, ymax) to set y-axis limits. If None, uses automatic limits.
    """
    fig, axis = plt.subplots(figsize=(12, 6), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = pyg.utils.degree(
        edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]    # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if sea_surface is not None:
        coastal_line = get_african_upwelling_zone(sea_surface, n_pixels=1)
        plt.imshow(coastal_line, origin="lower", cmap="binary")
    if title is not None:
        axis.set_title(title)
        #axis.set_title("")
        

    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)

    # Create directory if it doesn't exist
    # directory = os.path.join(save_dir, graph_type) if graph_type is not None else save_dir
    directory = save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    safe_title = title.replace(" ", "_") if title else "graph"
    filepath = os.path.join(directory, f"{safe_title}.svg")
    plt.savefig(filepath, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Figure saved in: {filepath}")

def plot_graph_networkx(graph, title=None, level=None, graph_type=None, plot_edges=False, save_dir="figures"):
    """
    Plots a NetworkX graph, showing the nodes filtered by the specified level, with their degrees represented by color.
    This function has the advantage of being able te select in a herarchical graph which level to plot.
        Parameters:
        - graph: networkx.Graph
        Graph to plot.
        - title: str, optional
        Graph title.
        - level: int, optional
        Graph level to filter nodes by (e.g., 0 for the first level).
        - graph_type: str, optional
        Graph type, to create directories and save the image.
        - plot_edges: bool, optional
        Whether or not to plot the edges of the graph.
        - save_dir: str, optional
        Specifies the directory to save the figure.
    """
    nodes = [node for node in graph.nodes if node[0] == level] if level is not None else graph.nodes
    
    node_degrees = dict(graph.degree(nodes))
    
    node_positions = networkx.get_node_attributes(graph, 'pos')
    
    fig, axis = plt.subplots(figsize=(24, 12), dpi=400)
    
    if plot_edges:
        edges = graph.edges(graph.nodes)
        edge_positions = [(node_positions[u], node_positions[v]) for u, v in edges]
    
        for (x1, y1), (x2, y2) in edge_positions:
            axis.plot([x1, x2], [y1, y2], color='black', lw=0.4, zorder=1)
    
    degrees = [node_degrees[node] for node in nodes]
    x_positions = [node_positions[node][0] for node in nodes]
    y_positions = [node_positions[node][1] for node in nodes]
    
    node_scatter = axis.scatter(x_positions, y_positions, c=degrees, edgecolors='k', alpha=0.7, cmap='viridis')
    
    plt.colorbar(node_scatter, aspect=50, label='nodes degree')
    
    if title is not None:
        axis.set_title(title)
    
    # directory = os.path.join(save_dir, graph_type) if graph_type is not None else save_dir
    directory = save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, f"{title}.svg")
    plt.savefig(filepath, format="svg")
    
    plt.close()

def compute_metrics(degree_counter):
    if not degree_counter:
        return {
            "min_degree": None,
            "max_degree": None,
            "average_degree": None,
            "median_degree": None,
            "percentile_25_degree": None,
            "percentile_75_degree": None,
            "IQR_degree": None,
            "count_degrees": {}
        }
    degrees = list(degree_counter.elements())
    return {
        "total_nodes": len(degrees),
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "average_degree": sum(degrees) / len(degrees),
        "median_degree": np.median(degrees),
        "percentile_25_degree": np.percentile(degrees, 25),
        "percentile_75_degree": np.percentile(degrees, 75),
        "IQR_degree": np.percentile(degrees, 75) - np.percentile(degrees, 25),
        "count_degrees": dict(degree_counter)
    }

def metrics_from_networkx_graph(graph, level_to_consider=None):
    """
    Extracts connection metrics from a NetworkX graph.

    If 'level_to_consider' is specified, nodes whose first element of 
    the identifier is equal to that level are filtered out.
    Otherwise, metrics are calculated for all nodes.
    """
    if level_to_consider is not None:
        filtered_nodes = [node for node in graph.nodes if node[0] == level_to_consider]
        degree_mapping = dict(graph.degree(filtered_nodes))
        degree_counter = Counter(degree_mapping.values())
        total_edges = graph.number_of_edges()
    else:
        degree_mapping = dict(graph.degree())
        degree_counter = Counter(degree_mapping.values())

    metrics = compute_metrics(degree_counter)
    total_edges = graph.number_of_edges() if level_to_consider is None else total_edges
    metrics["total_edges"] = total_edges
    return metrics

def plot_degree_distribution(degree_counter, title, graph_type=None, save_dir="figures"):
    """
    Graphs the degree distribution using a degree counter.

    Parameters:
    - degree_counter: Counter
        Degree counter.
    - title: str
        Chart title.
    - graph_type: str, optional
        Type of graph for saving the figure. Used to create a directory if it doesn't exist.
    - save_dir: str, optional
        Directory to save the figure.
    """
    degrees, counts = zip(*sorted(degree_counter.items()))
    plt.figure(figsize=(8, 6))
    plt.bar(degrees, counts, color='skyblue')
    plt.xlabel("Degree")
    plt.ylabel("Nodes count")

    if title is not None:
        plt.title(title)
    
    # directory = os.path.join(save_dir, graph_type) if graph_type is not None else save_dir
    directory = save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, f"{title}.svg")
    plt.savefig(filepath, format="svg")
    plt.close()

def create_metrics_table(sender_metrics, receiver_metrics, columns_names, metrics_to_avoid, title, graph_type,save_dir="figures"):
    """
    Create a table with the metrics of the sender and reciver graphs.
    Parameters:
    sender_metrics: dict
        Dictionary with the metrics of the sender graph.
    receiver_metrics: dict
        Dictionary with the metrics of the reciver graph.
    columns_names: list
        List with the names of the columns.
    metrics_to_avoid: list
        List with the names of the metrics to avoid.
    title: str
        Title of the table.
    graph_type: str
        Type of graph.
    save_dir: str
        Directory where to save the table.
    """
    metric_labels = [key for key in sender_metrics.keys() if key not in metrics_to_avoid]

    rows = metric_labels
    if receiver_metrics is None:
        data = [[sender_metrics[label]] for label in metric_labels]
        data = [[round(value, 2) for value in row] for row in data]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=data, 
                    rowLabels=rows, 
                    colLabels=columns_names,
                    loc='center',
                    cellLoc='center'
                    )
    else:
        data = [[sender_metrics[label], receiver_metrics[label]] for label in metric_labels]
        data = [[round(value, 2) for value in row] for row in data]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=data, 
                    rowLabels=rows, 
                    colLabels=columns_names, 
                    loc='center',
                    cellLoc='center')
    
    ax.set_title(title, fontsize=16)
    # directory = os.path.join(save_dir, graph_type) if graph_type is not None else save_dir
    directory = save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, f"{title}.svg")
    fig.tight_layout()
    plt.savefig(filepath, format="svg", bbox_inches='tight')
    #plt.show()

def count_nodes_in_masks(masks: dict, list_coords) -> dict:
    """
    Count how many nodes fall within each boolean mask.

    Parameters
    ----------
    masks : dict[str, np.ndarray(bool)]
        Dictionary mapping mask names to 2D boolean arrays of shape (300, 300).
        True indicates the cell is inside the region.

    list_coords : array-like of shape (N, 2)
        Sequence of (x, y) integer pixel coordinates for N nodes.

    Returns
    -------
    counts : dict[str, int]
        Dictionary mapping each mask name to the count of nodes that lie inside it.
    """
    coords = np.asarray(list_coords, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("list_coords must be an array of shape (N, 2) with (x, y) pairs.")

    xs, ys = coords[:, 0], coords[:, 1]
    counts = {}
    for name, mask in masks.items():

        in_bounds = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
        xs_in, ys_in = xs[in_bounds], ys[in_bounds]

        counts[name] = int(np.count_nonzero(mask[ys_in, xs_in]))

    return counts


def plot_counts_table(counts: dict, title: str = "Counts in Masks", filepath: str = None):
    """
    Generate a bar chart of node counts per mask with the exact number annotated above each bar.

    Parameters
    ----------
    counts : dict[str, int]
        Mapping from mask name to node count.

    title : str, optional
        Plot title. Default is "Counts in Masks".

    filepath : str, optional
        File path to save the figure. If None, display it instead.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(counts.keys(), counts.values())
    ax.set_ylabel("Number of Nodes")
    ax.set_title(title)
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha='center',
            va='bottom'
        )

    plt.tight_layout()

    if filepath:
        save_dir = os.path.dirname(filepath)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        plt.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()