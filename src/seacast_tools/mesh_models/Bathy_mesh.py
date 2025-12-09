import networkx
import numpy as np
import scipy.spatial
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from mesh_models.Non_uniform_mesh import NonUniformMesh
import mesh_models.mesh_connector as mc
import mesh_models.mesh_metrics as mm
import os
from mesh_models.density_distributions import generate_mixed_distribution, generate_basic_probability_distribution

class BathyMesh(NonUniformMesh):
    """
    Bathymetric mesh class for generating a 2D non-uniform graph using networkx.
    Inherits from NonUniformMesh class.
    """

    def __init__(self, xy, nx, land_mask, graph_type, root_path_to_save, bathymetry,seed=42, 
                fixed_total_nodes=False, total_nodes_for_each_level=None, distribution_type="mixed_sigmoid", 
                supplementary_masks=None):
        super().__init__(xy, nx, land_mask, graph_type, root_path_to_save, supplementary_masks)
        self.seed = seed
        self.bathymetry = bathymetry
        self.fixed_total_nodes = fixed_total_nodes
        self.total_nodes_for_each_level = total_nodes_for_each_level
        self.distribution_type = distribution_type
        self.supplementary_masks = supplementary_masks


    def generate_mesh_positions_using_bathymetry(self, 
        nx, ny, xm, xM, ym, yM, proportion_sea_nodes
    ):
    
        """
        Generate a list of coordinates (x,y) of the nodes to create a Graph with networkx.
        Uses bathymetry data to generate a probability distribution and select weighted positions.
        Also ensures that the corners are included so no information is lost in the corners of the mesh.
        Parameters:
            nx, ny: number of nodes in x and y directions (useful for calculating total nodes) (ints)
            xm, xM: limits in x (int)
            ym, yM: limits in y (int)
            bathymetry: 2D array (e.g., loaded from a .npy file) with depth values, where 0 is land and >0 is sea.
            proportion_sea_nodes: proportion of sea nodes to generate (float)
            land_mask: boolean mask indicating land (True) and sea (False) (2D npy array)
            total_nodes_for_each_level: dictionary with the total number of nodes to generate in each level, using the resolution
                (for this implementation, we will use nx as key) as a key (dict: {nx: total_nodes})
            fixed_total_nodes: if True, the total number of nodes is fixed to the value given by total_nodes (bool)
        """
        list_coords = []
        np.random.seed(self.seed)
        if self.fixed_total_nodes is True:
            # If fixed total nodes, use the value from the dictionary
            total_nodes = self.total_nodes_for_each_level[nx]
        else:
            total_nodes = int((nx * ny) * proportion_sea_nodes)

        x = np.linspace(xm, xM, self.bathymetry.shape[0])
        y = np.linspace(ym, yM, self.bathymetry.shape[1])
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        print("distribution type: ", self.distribution_type)
        if self.distribution_type == "mixed_sigmoid":
            p_mix = generate_mixed_distribution(self.bathymetry, alpha=0.7)
        elif self.distribution_type == "base":
            p_mix = generate_basic_probability_distribution(self.bathymetry)
        else:
            print("no distribution type selected, using base")
            p_mix = generate_basic_probability_distribution(self.bathymetry)
        flat_idx = np.arange(self.bathymetry.size)
        chosen = np.random.choice(flat_idx, size=total_nodes, replace=False, p=p_mix.ravel())
        pts = np.column_stack([X.ravel()[chosen], Y.ravel()[chosen]])

        self.generate_ocean_corner_nodes(self.land_mask, x, y, pts, list_coords)
        # Grid generation as a placeholder for coordinates with the shape of our domain
        if self.fixed_total_nodes is True:
            #append the coordinates of the points to the list_coords until the total number of nodes is reached
            for pt in pts:
                if len(list_coords) < total_nodes:
                    list_coords.append(tuple(pt))
                else:
                    break
        else:
            for pt in pts:
                list_coords.append(tuple(pt))
        
        return list_coords

    def generate_ocean_corner_nodes(self, land_mask, x, y, pts, list_coords):
        corner_dist_threshold = 5.0

        tree = KDTree(pts)

        ocean_mask = ~land_mask
        # distance to land
        dist_to_land = distance_transform_edt(ocean_mask)

        corners_idx = self.get_ocean_corners(land_mask.T, inverted=True)

        window = 2
        coastal_margin = np.sqrt(0.5)

        # for each corner, check if it is too close to the coast
        for name, (ci, cj) in corners_idx.items():
            cx, cy = x[ci], y[cj]
            dist, _ = tree.query((cx, cy), k=1)

            if dist > corner_dist_threshold:
                # limits of the search window
                i0, i1 = max(0, ci - window), min(land_mask.shape[0] - 1, ci + window)
                j0, j1 = max(0, cj - window), min(land_mask.shape[1] - 1, cj + window)

                #neighboring points in the search window that are ocean and far enough from the land
                neigh_idxs = [
                    (ii, jj)
                    for ii in range(i0, i1 + 1)
                    for jj in range(j0, j1 + 1)
                    if ocean_mask[ii, jj] and dist_to_land[ii, jj] > coastal_margin
                ]

                if neigh_idxs:
                    #pick one
                    ii, jj = neigh_idxs[np.random.randint(len(neigh_idxs))]
                    nx_coord, ny_coord = x[ii], y[jj]
                else:
                    #if no one has been chosen, use the corner
                    nx_coord, ny_coord = cx, cy

                # add and update the coordinates
                list_coords.append((nx_coord, ny_coord))
                pts = np.vstack([pts, [nx_coord, ny_coord]])
                tree = KDTree(pts)

    def get_ocean_corners(self, mask, inverted=False):
        """
        For a 2D Boolean mask from a numpy array, returns the four corners of the ocean area.

        Parameters:
        mask - 2D Boolean array (npy)
        inverted — if True, interprets mask==False as ocean (and True as land)
        """
        # if land_mask provided, invert it to get ocean_mask
        ocean_mask = (~mask) if inverted else mask
        
        # index to all ocean points
        rows, cols = np.where(ocean_mask)
        if rows.size == 0:
            raise ValueError("mask must contain at least one ocean point")
        
        # Bounding‑box of the ocean points
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        targets = {
            'bottom_left':  (min_r, min_c),
            'bottom_right': (min_r, max_c),
            'top_left':     (max_r, min_c),
            'top_right':    (max_r, max_c),
        }
        
        # search for the nearest ocean point to each corner of the domain
        nearest = {}
        for name, (tr, tc) in targets.items():
            # calculate the distance to each ocean point
            d2 = (rows - tr)**2 + (cols - tc)**2
            idx = np.argmin(d2)
            nearest[name] = (rows[idx], cols[idx])
        return nearest

    def mk_2d_non_uniform_graph(self, n_nodes_x, n_nodes_y,*args, **kwargs):
        """
        Create a 2D non-uniform graph using networkx where:
        - The function takes a mesh grid defined by the xy coordinates and the number of nodes in x and y direction.
        - The function generates positions for the nodes inside the mesh.
        - The function ensures that the positions of the nodes are not too close to each other by setting a minimum distance between them.
        - The function also ensures that the positions of the nodes are within the limits of the mesh.
        - The function ensures that the nodes are not placed on land by checking the land mask.
        - The function adds edges to the graph based on the Delaunay triangulation of the nodes.

        Params:
        xy: mesh grid coordinates
        nx, ny: number of nodes in x and y direction
        land_mask: mask of land nodes
        sea_depth: depth of the sea(bathymetry)

        returns:
        - dg: directed graph with nodes and edges
        """
        xm, xM = np.amin(self.xy[0][0, :]), np.amax(self.xy[0][0, :])
        ym, yM = np.amin(self.xy[1][:, 0]), np.amax(self.xy[1][:, 0])
        sea_nodes = np.sum(self.land_mask == False)
        land_nodes = np.sum(self.land_mask == True)
        proportion_sea_nodes = round(sea_nodes / (sea_nodes + land_nodes), 2)
        list_coords = self.generate_mesh_positions_using_bathymetry(n_nodes_x, n_nodes_y, xm, xM, ym, yM, 
                                                                    proportion_sea_nodes= proportion_sea_nodes)
        counts = mm.count_nodes_in_masks(self.supplementary_masks, list_coords)
        title = f"total nodes for each mask in level {n_nodes_x}.svg"
        filepath = os.path.join(self.path_to_save_figures, self.graph_type, title)
        mm.plot_counts_table(counts, title, filepath)

        g = networkx.Graph()
        for coord in list_coords:
            g.add_node(coord)

        # kdtree for nearest neighbor search of land nodes
        land_points = np.argwhere(self.land_mask.T).astype(np.float32)
        land_kdtree = scipy.spatial.KDTree(land_points)

        # add nodes excluding land
        for node in list(g.nodes):
            node_pos = np.array(node, dtype=np.float32)
            dist, _ = land_kdtree.query(node_pos, k=1)
            if self.fixed_total_nodes is True:
                if dist < np.sqrt(0.5):
                    # Relocate the node until a valid (non-land) position is found
                    max_attempts = 1000
                    attempt = 0
                    while attempt < max_attempts:
                        attempt += 1

                        if self.distribution_type == "mixed_sigmoid":
                            p = generate_mixed_distribution(self.bathymetry, alpha=0.7)
                        elif self.distribution_type == "base":
                            p = generate_basic_probability_distribution(self.bathymetry)
                        else:
                            print("No distribution type selected, using base")
                            p = generate_basic_probability_distribution(self.bathymetry)

                        # Normalize and flatten the probability distribution
                        p_flat = p.ravel() / np.sum(p)
                        flat_idx = np.arange(self.bathymetry.size)
                        chosen_idx = np.random.choice(flat_idx, size=1, p=p_flat)[0]

                        # Convert flat index back to 2D indices
                        i, j = np.unravel_index(chosen_idx, self.bathymetry.shape)

                        # Get new coordinates from mesh grid
                        x_new = np.linspace(xm, xM, self.bathymetry.shape[0])[i]
                        y_new = np.linspace(ym, yM, self.bathymetry.shape[1])[j]
                        pos_new = np.array([x_new.item(), y_new.item()], dtype=np.float32)

                        dist_new, _ = land_kdtree.query(pos_new, k=1)
                        if dist_new >= np.sqrt(0.5):
                            g.nodes[node]['pos'] = pos_new
                            break
                    else:
                        print(f"No valid new position found for node {node} after {max_attempts} attempts.")
                        g.nodes[node]['pos'] = node_pos  # keep original position as fallback
                else:
                    g.nodes[node]["pos"] = node_pos

            else:
                if dist < np.sqrt(0.5):
                    g.remove_node(node)
                else:
                    g.nodes[node]["pos"] = node_pos

        mc.add_Delaunay_edges(g)

        # remove edges that goes across land
        for u, v in list(g.edges()):
            if mc.crosses_land(g.nodes[u]["pos"], g.nodes[v]["pos"], self.land_mask):
                g.remove_edge(u, v)

        # turn into directed graph
        dg = networkx.DiGraph(g)

        # add node data
        for u, v in g.edges():
            d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
            dg.edges[u, v]["len"] = d
            dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
            dg.add_edge(v, u)
            dg.edges[v, u]["len"] = d
            dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

        # add self edge if needed
        for v, degree in list(dg.degree()):
            if degree <= 1:
                dg.add_edge(v, v, len=0, vdiff=np.array([0, 0]))

        return dg

    def add_down_edges(self, G_down, v_to_list, v_from_list, kdt_m, k_neighboors,*args, **kwargs):
        mc.add_edges_based_in_k_neighboors(G_down, v_from_list, v_to_list, kdt_m, self.land_mask, k_neighboors=k_neighboors)

    def establish_edges_g2m(self, graph, sender_nodes_list, receiver_nodes_list, required_connections,*args, **kwargs):
        receiver_nodes_list = [node for node in receiver_nodes_list if node[0] == 0]
        mc.establish_n_edges_g2m(graph, sender_nodes_list, receiver_nodes_list, required_connections=required_connections)

    def establish_edges_m2g(self, graph, *args, **kwargs):
        G_m2g = mc.create_reversed_graph(graph)
        return G_m2g
    
    