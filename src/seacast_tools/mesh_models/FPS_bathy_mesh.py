import networkx
import numpy as np
import scipy.spatial
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from mesh_models.FPS_mesh import FPSMesh
import mesh_models.mesh_connector as mc
import mesh_models.mesh_metrics as mm
import os

class FPSBathyMesh(FPSMesh):
    def __init__(self, xy, nx, land_mask, graph_type, root_path_to_save, 
                 bathymetry ,seed=42, fixed_total_nodes=False,
                 total_nodes_for_each_level=None, supplementary_masks=None):
        super().__init__(xy, nx, land_mask, graph_type, root_path_to_save, supplementary_masks)
        self.seed = seed
        self.fixed_total_nodes = fixed_total_nodes
        self.total_nodes_for_each_level = total_nodes_for_each_level
        self.supplementary_masks = supplementary_masks
        self.bathymetry = bathymetry

    def mk_2d_non_uniform_graph(self, nx, ny, *args, **kwargs):
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

        returns:
        - dg: directed graph with nodes and edges
        """

        xm, xM = np.amin(self.xy[0][0, :]), np.amax(self.xy[0][0, :])
        ym, yM = np.amin(self.xy[1][:, 0]), np.amax(self.xy[1][:, 0])

        sea_nodes = np.sum(self.land_mask == False)
        land_nodes = np.sum(self.land_mask == True)
        proportion_sea_nodes = round(sea_nodes / (sea_nodes + land_nodes), 2)
        #list_coords = vertex_clustering(nx, ny, xm, xM, ym, yM, self.bathymetry, proportion_sea_nodes=proportion_sea_nodes, land_mask=land_mask, total_nodes_for_each_level=total_nodes_for_each_level, fixed_total_nodes=fixed_total_nodes)
        list_coords = self.vertex_clustering_decimation(nx, ny,
                                                        xm, xM, ym, yM, 
                                                        proportion_sea_nodes=proportion_sea_nodes
                                                   )
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

    def vertex_clustering_decimation(self, nx, ny, 
                                     xm, xM, ym, yM,
                                     proportion_sea_nodes,
    ) -> list:
        """
        Generate nodes by using farthest point sampling as a way of decimation.

        Args:
            nx, ny: number of cells in x and y.
            xm, xM / ym, yM: limits of the mesh.
            bathymetry: 2D array (>0 = sea).
            proportion_sea_nodes: proporción de mar a muestrear si no es fijo.

        Returns:
            List of thuples (x, y) with the selected points.
        """
        np.random.seed(self.seed)
        list_coords = []
        x = np.linspace(xm, xM, self.bathymetry.shape[0])
        y = np.linspace(ym, yM, self.bathymetry.shape[1])
        X, Y = np.meshgrid(x, y, indexing='xy')
        coords = np.vstack((X.ravel(), Y.ravel())).T

        #filter nodes that are not in the sea
        sea_mask = (~self.land_mask.ravel()) & (self.bathymetry.ravel() > 0)
        sea_coords = coords[sea_mask]
        n_sea = sea_coords.shape[0]
        if n_sea == 0:
            return []

        # set the number of nodes to sample
        if self.fixed_total_nodes:
            total = self.total_nodes_for_each_level.get(nx, None)
            if total is None:
                total = int((nx * ny) * proportion_sea_nodes)
        else:
            total = int((nx * ny) * proportion_sea_nodes)

        # sample the points using farthest point sampling
        total = max(0, min(total, n_sea))
        if total == 0:
            return []
        sampled = self.farthest_point_sampling(sea_coords, k=total)
        list_coords = [tuple(pt) for pt in sampled]

        return list_coords

    def farthest_point_sampling(self, points: np.ndarray,
                                     weights: np.ndarray, k: int) -> np.ndarray:
        n = points.shape[0]
        if k <= 0 or n == 0:
            return np.empty((0,2))

        # Choose a point with probability proportional to the weights
        idx0 = np.random.choice(n, p=weights/weights.sum())
        selected = [idx0]
        dist = np.linalg.norm(points - points[idx0], axis=1)

        for _ in range(1, min(k, n)):
            score     = dist * weights
            next_idx  = int(np.argmax(score))
            selected.append(next_idx)
            new_dist  = np.linalg.norm(points - points[next_idx], axis=1)
            dist      = np.minimum(dist, new_dist)

        return points[selected]
