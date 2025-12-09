"""
This module contains different types of mesh edges creation
functions, where we can create different approaches for the 
interconnection logic between mesh levels, G2M and M2G.
"""
import numpy as np
import scipy.spatial
import skimage.draw


def crosses_land(node1, node2, land_mask, threshold=8):
    x1, y1 = node1
    x2, y2 = node2
    rr, cc = skimage.draw.line(round(y1), round(x1), round(y2), round(x2))

    return np.sum(land_mask[rr, cc]) >= threshold

def add_features(graph, receiver_node, sender_node):
    """
    Adds features to the graph edges.
    Features are the distance between the nodes and the vector difference between the nodes where: 
    - The distance is calculated using the Euclidean distasnce formula.
    - The vector difference is calculated by subtracting the position of one node from the other.
    - The features are added to the graph edges as attributes.
    - The attributes are named "len" and "vdiff".
    - The "len" attribute is the distance between the nodes.
    - The "vdiff" attribute is the vector difference between the nodes.
    """
    d = np.sqrt(np.sum((graph.nodes[sender_node]["pos"] - graph.nodes[receiver_node]["pos"]) ** 2))
    graph.edges[sender_node, receiver_node]["len"] = d
    graph.edges[sender_node, receiver_node]["vdiff"] = (
                        graph.nodes[sender_node]["pos"] - graph.nodes[receiver_node]["pos"]
                        )

def establish_edges_based_in_proportion(graph, receiver_nodes_list, sender_nodes_list, proportion):
    """
    Establish edges between mesh nodes and grid nodes based on a given proportion where:
    - The proportion is used to determine the radius of connection between mesh nodes and grid nodes.
    - The function connects the mesh nodes to the grid nodes within the radius determined by the proportion.
    - If the grid nodes are not connected to any mesh node, it connects them to the nearest mesh node.

    Params:
    graph: networkx.Graph
        Graph in which the edges will be added.
    receiver_nodes_list: list
        List of nodes, where the mesh nodes are those whose first element of the tuple is 0.
    sender_nodes_list: list
        List of grid nodes (expected to have the form (1000, x, y)).
    proportion: float
        Proportion used to determine the radius of connection between mesh nodes and grid nodes.
        The radius is calculated as the square root of the proportion divided by 2.

    returns:
    - None: The function modifies the graph in place by adding edges between the nodes.
    """
    sender_nodes_connected = {node: 0 for node in sender_nodes_list}
    #receiver_nodes = [node for node in receiver_nodes_list if node[0] == 0]
    receiver_nodes = [node for node in receiver_nodes_list]
    receiver_node_connected = {node: 0 for node in receiver_nodes}
    radio = np.sqrt(proportion)/2
    #We round the radio to next integer
    radio = int(np.ceil(radio))
    grid_positions = np.array([np.array(graph.nodes[node]["pos"]) for node in sender_nodes_list])
    grid_tree = scipy.spatial.KDTree(grid_positions)
    connect_receiver_nodes_to_sender_nodes_within_radius(graph, sender_nodes_list, sender_nodes_connected, 
                                                   receiver_nodes, receiver_node_connected, radio, 
                                                   grid_tree)

    connect_unlinked_sender_nodes(graph, sender_nodes_connected, receiver_nodes)


def connect_receiver_nodes_to_sender_nodes_within_radius(graph, sender_nodes_list, sender_nodes_connected, receiver_nodes, 
                                                   receiver_node_connected, radio, grid_tree):
    """
    Connects mesh nodes to grid nodes within a given radius.
    It iterates through the mesh nodes and checks if there are grid nodes within the radius.
    If there are, it creates edges between the mesh nodes and the grid nodes.
    """
    for receiver_node in receiver_nodes:
        mesh_pos = np.array(graph.nodes[receiver_node]["pos"])
        _, indices = grid_tree.query(mesh_pos, k=len(sender_nodes_list), distance_upper_bound=radio)
        for idx in indices:
            if idx == len(sender_nodes_list):
                continue
            sender_node = sender_nodes_list[idx]
            if not graph.has_edge(sender_node, receiver_node):
                graph.add_edge(sender_node, receiver_node)
                add_features(graph, receiver_node, sender_node)
                sender_nodes_connected[sender_node] += 1
                receiver_node_connected[receiver_node] += 1

def connect_unlinked_sender_nodes(graph, sender_nodes_connected, receiver_nodes):
    """
    Connects unlinked grid nodes to the nearest mesh node to ensure that all grid nodes are connected.
    It iterates through the grid nodes and checks if they are connected to any mesh node. If not, it finds the nearest mesh node and creates an edge between them.
    The distance taken into account in order to connect the nodes is the Euclidean distance.

    Parameters:
    graph : networkx.Graph
        Graph in which the edges will be added.
    sender_nodes_connected : dict
        Dictionary with the grid nodes as keys and the number of connections as values.
    receiver_nodes : list
        List of nodes that are considered mesh nodes(or receiver nodes in general terms).
    """
    for sender_node, count in sender_nodes_connected.items():
        if count == 0:
            # Find the nearest mesh node
            best_receiver_node, best_distance = None, float('inf')
            grid_pos = graph.nodes[sender_node]["pos"]
            for receiver_node in receiver_nodes:
                mesh_pos = graph.nodes[receiver_node]["pos"]
                d = np.linalg.norm(np.array(grid_pos) - np.array(mesh_pos))
                if d < best_distance:
                    best_distance = d
                    best_receiver_node = receiver_node
            if best_receiver_node is not None and not graph.has_edge(sender_node, best_receiver_node):
                graph.add_edge(sender_node, best_receiver_node)
                add_features(graph, best_receiver_node, sender_node)
                sender_nodes_connected[sender_node] += 1

def connect_unlinked_receiver_nodes(
    graph,
    receiver_nodes_connected: dict,
    sender_nodes_list: list
):
    """
    Connect each unlinked receiver node to its nearest sender node.

    This ensures that every receiver node has at least one incoming edge
    from a sender. It computes the Euclidean distance between the receiver's
    position and all sender positions, then adds the shortest edge if missing.

    Parameters:
        graph (networkx.Graph): The graph in which to add edges.
        receiver_nodes_connected (dict): Mapping from receiver node IDs to the
            current number of incoming edges they have.
        sender_nodes_list (list): List of sender node IDs to consider as potential sources.
    """
    for v, count in receiver_nodes_connected.items():
        if count == 0:
            best_u = None
            best_dist = float('inf')
            pos_v = np.array(graph.nodes[v]["pos"])

            # Find the nearest sender
            for u in sender_nodes_list:
                pos_u = np.array(graph.nodes[u]["pos"])
                d = np.linalg.norm(pos_u - pos_v)
                if d < best_dist:
                    best_dist = d
                    best_u = u

            # Add the edge if it doesn't already exist
            if best_u is not None and not graph.has_edge(best_u, v):
                graph.add_edge(best_u, v)
                # Optionally, add edge attributes (length, vector difference)
                delta = graph.nodes[best_u]["pos"] - graph.nodes[v]["pos"]
                graph.edges[best_u, v]["len"] = best_dist
                graph.edges[best_u, v]["vdiff"] = delta
                receiver_nodes_connected[v] += 1


def establish_n_edges_g2m(graph, sender_nodes_list, receiver_nodes_list, required_connections):
    """
    Establishes connections between grid nodes and mesh nodes in the graph.

    Parameters:
        graph : networkx.Graph
            Graph in which the edges will be added.
        receiver_nodes_list : list
            List of nodes, where mesh nodes are considered to be those whose first element
            in the tuple is 0.
        sender_nodes_list : list
            List of grid nodes (expected to be in the form (1000, x, y)).
        xy : any
            Unused variable in this version (could be removed or used for further optimizations).
        required_connections : int
            Number of connections to establish for each node (both grid and mesh).
        coef : float, optional
            Coefficient used to determine whether the distance between nodes is large enough 
            to avoid creating a connection. If this feature is not needed, set coef=inf.

    Function Behavior:
        1. For each grid node, finds its `required_connections` closest mesh nodes and creates the connections.
        2. Ensures that each mesh node has at least `required_connections` connections by adding the missing 
        ones to the closest grid nodes that are not yet connected.

    Returns:
        None
            The function modifies the graph in place by adding edges between nodes.
    """
    sender_nodes_connected = {node: 0 for node in sender_nodes_list}
    #receiver_nodes = [node for node in receiver_nodes_list if node[0] == 0]
    receiver_nodes = [node for node in receiver_nodes_list]
    receiver_node_connected = {node: 0 for node in receiver_nodes}

    mesh_positions = np.array([np.array(graph.nodes[node]["pos"]) for node in receiver_nodes])
    
    for sender_node in sender_nodes_list:
        grid_pos = np.array(graph.nodes[sender_node]["pos"])
        dists = np.linalg.norm(mesh_positions - grid_pos, axis=1)
        sorted_indices = np.argsort(dists)
        for idx in sorted_indices[:required_connections]:
            receiver_node = receiver_nodes[idx]
            if not graph.has_edge(sender_node, receiver_node):
                graph.add_edge(sender_node, receiver_node)
                add_features(graph, receiver_node, sender_node)
                sender_nodes_connected[sender_node] += 1
                receiver_node_connected[receiver_node] += 1

    grid_positions = np.array([np.array(graph.nodes[node]["pos"]) for node in sender_nodes_list])
    grid_tree = scipy.spatial.KDTree(grid_positions)

    for receiver_node in receiver_nodes:
        while receiver_node_connected[receiver_node] < required_connections:
            mesh_pos = np.array(graph.nodes[receiver_node]["pos"])
            dists, indices = grid_tree.query(mesh_pos, k=len(sender_nodes_list))
            found = False
            for idx in indices:
                candidate = sender_nodes_list[idx]
                if not graph.has_edge(candidate, receiver_node):
                    graph.add_edge(candidate, receiver_node)
                    add_features(graph, receiver_node, candidate)
                    sender_nodes_connected[candidate] += 1
                    receiver_node_connected[receiver_node] += 1
                    found = True
                    break
            if not found:
                break

def add_edges_based_in_k_neighboors(graph, sender_nodes_list, receiver_nodes_list, kdt_m, land_mask, k_neighboors=16):
    """
    Adds edges from each 'receiver' node to its k nearest neighbors in 'sender'.
    
    :param graph: NetworkX DiGraph containing both receiver and sender nodes.
    :param sender_nodes_list: list of node IDs from the higher-level graph.
    :param receiver_nodes_list: list of node IDs from the lower-level graph.
    :param kdt_m: KDTree built over the positions of sender_nodes_list.
    :param land_mask: land mask used to discard invalid crossings.
    :param k_neighboors: number of neighbors to search.
    """
    sender_nodes_connected = {node: 0 for node in sender_nodes_list}
    receiver_nodes_connected = {node: 0 for node in receiver_nodes_list}

    for v in receiver_nodes_list:
        # find k neighbors (returns scalar if k=1, array if k>1)
        neigh_idx = kdt_m.query(graph.nodes[v]["pos"], k=k_neighboors)[1]
        # ensure we always work with a 1-D array
        neigh_idx = np.atleast_1d(neigh_idx)

        for idx in neigh_idx:
            u = sender_nodes_list[idx]
            # discard land crossings
            if not crosses_land(
                graph.nodes[u]["pos"],
                graph.nodes[v]["pos"],
                land_mask=land_mask
            ):
                graph.add_edge(u, v)
                sender_nodes_connected[u] += 1
                receiver_nodes_connected[v] += 1
                # edge metrics
                d = np.linalg.norm(
                    graph.nodes[u]["pos"] - graph.nodes[v]["pos"]
                )
                graph.edges[u, v]["len"] = d
                graph.edges[u, v]["vdiff"] = (
                    graph.nodes[u]["pos"] - graph.nodes[v]["pos"]
                )

    # ensure at least one connection per node
    connect_unlinked_sender_nodes(graph, sender_nodes_connected, receiver_nodes_list)
    connect_unlinked_receiver_nodes(graph, receiver_nodes_connected, sender_nodes_list)
                    
            


def establish_edge_with_distances_range(graph, sender_nodes_list, receiver_nodes_list, distances,xy, min_neighboors=3, max_neighboors=16):
    """
    Establish the edges between the nodes of the mesh and the nodes of the grid
    according to the distances between the nodes of the mesh and the nodes of the grid.

    Parameters

    graph: networkx graph
        Graph where the edges will be established (Usually G2M Graph)
    
    distances: list
        List with the min value, the percentiles and the max value
        of the distances between the nodes of the mesh
    
    receiver_nodes_list: networkx graph
        Graph with the nodes of the lowest level of the mesh (with highest resolition)
    
    DM_SCALE: float
        Scale factor to establish the distance range between the nodes of the mesh and the nodes of the grid

    sender_nodes_list: list
        List with the nodes of the grid
    
    """
    # dictionary with node as key and number of connections as value
    sender_nodes_connected = {}
    vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] for node in sender_nodes_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # while the number of nodes in sender_nodes_connected is not equal to the number of nodes in sender_nodes_list
    while len(sender_nodes_connected) < len(sender_nodes_list):
        for receiver_node in receiver_nodes_list:
            n_connections = 0
            for distance in distances:
                # find k nearest neighbours (index to vg_xy)
                neigh_idx = kdt_g.query_ball_point(receiver_nodes_list[receiver_node]["pos"], distance)
                for idx in neigh_idx:
                    sender_node = sender_nodes_list[idx]
                    #if n_connections == max_neighboors: #and distance != distances[0]s
                    #    break
                    graph.add_edge(sender_node, receiver_node)
                    add_features(graph, receiver_node, sender_node)
                    n_connections += 1
                    #sender_nodes_connected[sender_node]+=1
                    if n_connections >= max_neighboors:
                        break
                #add node to the dictionary
                    sender_nodes_connected[sender_node] = n_connections
                if n_connections >= min_neighboors:
                    break

        if n_connections < min_neighboors:
            # find min_neighbors nearest neighbours
            neigh_idx = kdt_g.query(receiver_nodes_list[receiver_node]["pos"], min_neighboors)[1]
            for idx in neigh_idx:
                sender_node = sender_nodes_list[idx]
                graph.add_edge(sender_node, receiver_node)
                add_features(graph, receiver_node, sender_node)
                n_connections += 1
                sender_nodes_connected[sender_node] = n_connections

def add_Delaunay_edges(g):
    """
    Adds Delaunay edges to the graph.
    This function is intended to be used for creating edges between nodes in a mesh or grid of the same level 
    in it's creation process. It uses the Delaunay triangulation to determine the edges.

    parameters:
    g: networkx.Graph
        Graph where the edges will be added.
        The graph should have nodes with a 'pos' attribute that contains the coordinates of the node.
    
    """
    g.remove_edges_from(list(g.edges()))
    
    nodes_list = list(g.nodes())
    points = np.array([g.nodes[node]["pos"] for node in nodes_list])
    
    tri = scipy.spatial.Delaunay(points)
    
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                node1 = nodes_list[simplex[i]]
                node2 = nodes_list[simplex[j]]
                if g.has_node(node1) and g.has_node(node2):
                    g.add_edge(node1, node2)

def create_reversed_graph(graph):
    """
    This function receives a directed graph from networkx and returns a new graph
    with the same connections but with the direction of all edges reversed.
    """
    reversed_graph = graph.reverse(copy=True)
    return reversed_graph