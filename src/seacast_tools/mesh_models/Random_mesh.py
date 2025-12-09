import networkx
import matplotlib.pyplot as plt
import numpy as np
import os
import torch_geometric as pyg
import matplotlib
import scipy.spatial
from mesh_models.Non_uniform_mesh import NonUniformMesh
import mesh_models.mesh_connector as mc
import mesh_models.mesh_metrics as mm
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt

class RandomMesh(NonUniformMesh):

    def __init__(self, xy, nx, land_mask, graph_type, root_path_to_save, 
                 seed=42, fixed_total_nodes=False,
                 total_nodes_for_each_level=None, supplementary_masks=None):
        super().__init__(xy, nx, land_mask, graph_type, root_path_to_save, supplementary_masks)
        self.seed = seed
        self.fixed_total_nodes = fixed_total_nodes
        self.total_nodes_for_each_level = total_nodes_for_each_level
        self.supplementary_masks = supplementary_masks

    def mk_2d_non_uniform_graph(self, n_nodes_x, n_nodes_y, *args, **kwargs):
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
        lower_offset_bound: lower bound for random offsets
        upper_offset_bound: upper bound for random offsets
        mode: mode of generation, either "random_offsets" or "random"

        returns:
        - dg: directed graph with nodes and edges
        """

        xm, xM = np.amin(self.xy[0][0, :]), np.amax(self.xy[0][0, :])
        ym, yM = np.amin(self.xy[1][:, 0]), np.amax(self.xy[1][:, 0])


        list_coords = self.generate_random_nodes_positions(n_nodes_x, n_nodes_y, xm, xM, ym, yM, self.seed, 
                                                            fixed_total_nodes=self.fixed_total_nodes,
                                                            total_nodes_for_each_level=self.total_nodes_for_each_level)

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
                    while True:
                        x_new = np.random.uniform(xm, xM)
                        y_new = np.random.uniform(ym, yM)
                        pos_new = np.array([x_new, y_new])
                        dist_new, _ = land_kdtree.query(pos_new, k=1)
                        if dist_new >= np.sqrt(0.5):
                            g.nodes[node]['pos'] = (x_new, y_new)
                            break
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

    def secure_positions_within_the_limit(self, xm, xM, ym, yM, dx, dy, lx, ly):
        lx[0] = xm + dx / 2
        lx[-1] = xM - dx / 2

        for i in range(len(lx)):    
            if lx[i] < xm + dx / 2:
                lx[i] = xm + dx / 2
            if lx[i] > xM - dx / 2:
                lx[i] = xM - dx / 2
        
        ly[0] = ym + dy / 2
        ly[-1] = yM - dy / 2

        for i in range(len(ly)):
            if ly[i] < ym + dy / 2:
                ly[i] = ym + dy / 2
            if ly[i] > yM - dy / 2:
                ly[i] = yM - dy / 2
    
    def generate_random_nodes_positions(self, nx, ny, xm, xM, ym, yM, seed,
                                        fixed_total_nodes=False, 
                                        total_nodes_for_each_level=None):
        """ 
        Generate random nodes positions inside the mesh

        Returns:
            list of coordinates (x, y) of the nodes to create a Graph with networkx
        """
        np.random.seed(seed)
        if fixed_total_nodes is True:
            list_coords = []
            ocean_mask = ~self.land_mask
            total_nodes = total_nodes_for_each_level[nx]
            N = nx*ny
            x = np.linspace(xm, xM, ocean_mask.shape[0])
            y = np.linspace(ym, yM, ocean_mask.shape[1])
            X, Y = np.meshgrid(x, y, indexing='xy')

            p = self.generate_uniform_distribution(ocean_mask, seed=seed)
            flat_idx = np.arange(ocean_mask.size)
            chosen = np.random.choice(flat_idx, size=N, replace=False, p=p.ravel())
            pts = np.column_stack([X.ravel()[chosen], Y.ravel()[chosen]])

            self.generate_ocean_corner_nodes(x, y, pts, list_coords)
            print(f"Number of points generated: {len(pts)}")
            for pt in pts:
                if len(list_coords) < total_nodes:
                    list_coords.append(tuple(pt))
                else:
                    break
        else:
            N = nx * ny 
            lx = np.random.uniform(xm, xM, N)
            ly = np.random.uniform(ym, yM, N)
            self.secure_positions_within_the_limit(xm, xM, ym, yM, (xM - xm) / nx, (yM - ym) / ny, lx, ly)
            list_coords = [(x, y) for x, y in zip(lx, ly)]
        return list_coords
    
    def generate_uniform_distribution(self, ocean_mask, seed=42):
        """
        Generate a uniform distribution of points in the ocean mask.

        Args:
            ocean_mask (numpy.ndarray): The ocean mask.
            N (int): The number of points to generate.
            seed (int): The random seed.

        Returns:
            numpy.ndarray: The generated points.
        """
        np.random.seed(seed)
        dist_to_land = distance_transform_edt(ocean_mask)
        coastal_margin = np.sqrt(0.6) 
        valid = (ocean_mask) & (dist_to_land >= coastal_margin)

        p = np.zeros_like(dist_to_land, dtype=float)
        p[valid] = np.random.uniform(0, 1, size=np.sum(valid))
        p /= p.sum()
        return p

    def generate_ocean_corner_nodes(self, x, y, pts, list_coords):
        corner_dist_threshold = 5.0

        tree = KDTree(pts)

        ocean_mask = ~self.land_mask
        # Each ocean cell gets the distance to the nearest land
        dist_to_land = distance_transform_edt(ocean_mask)

        corners_idx = self.get_ocean_corners(self.land_mask.T, inverted=True)

        # Window parameters and minimum distance to the coast
        window = 2
        coastal_margin = np.sqrt(0.5)

        # For each corner, check and add a valid neighbor
        for name, (ci, cj) in corners_idx.items():
            cx, cy = x[ci], y[cj]
            dist, _ = tree.query((cx, cy), k=1)

            if dist > corner_dist_threshold:
                # Limit the area according to the window
                i0, i1 = max(0, ci - window), min(self.land_mask.shape[0] - 1, ci + window)
                j0, j1 = max(0, cj - window), min(self.land_mask.shape[1] - 1, cj + window)

                # neighbors that are ocean and far enough from the coast
                neigh_idxs = [
                    (ii, jj)
                    for ii in range(i0, i1 + 1)
                    for jj in range(j0, j1 + 1)
                    if ocean_mask[ii, jj] and dist_to_land[ii, jj] > coastal_margin
                ]

                if neigh_idxs:
                    # Randomly choose one
                    ii, jj = neigh_idxs[np.random.randint(len(neigh_idxs))]
                    nx_coord, ny_coord = x[ii], y[jj]
                else:
                    # If there are none far enough, revert to the original corner
                    nx_coord, ny_coord = cx, cy

                # Add and update the KDTree
                list_coords.append((nx_coord, ny_coord))
                pts = np.vstack([pts, [nx_coord, ny_coord]])
                tree = KDTree(pts)

    def get_ocean_corners(self, mask, inverted=False):
        """
        For a 2D boolean mask, returns the 4 corners of the ocean area.
        
        Parameters:
        mask      — 2D boolean array
        inverted  — if True, interprets mask==False as ocean (and True as land)
        """
        # Adjust the mask so that 'ocean_mask' is True where there is ocean
        ocean_mask = (~mask) if inverted else mask

        # Indices of all ocean cells
        rows, cols = np.where(ocean_mask)
        if rows.size == 0:
            raise ValueError("The mask contains no ocean points")

        # Ocean bounding box
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        targets = {
            'bottom_left':  (min_r, min_c),
            'bottom_right': (min_r, max_c),
            'top_left':     (max_r, min_c),
            'top_right':    (max_r, max_c),
        }

        # For each vertex of the bounding box, find the nearest True cell
        nearest = {}
        for name, (tr, tc) in targets.items():
            # Compute squared distance to each ocean point
            d2 = (rows - tr)**2 + (cols - tc)**2
            idx = np.argmin(d2)
            nearest[name] = (rows[idx], cols[idx])
        return nearest