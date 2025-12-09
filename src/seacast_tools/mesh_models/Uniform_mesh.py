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
from torch_geometric.utils.convert import from_networkx

class UniformMesh(NonUniformMesh):
    """
    Uniform mesh class for generating a 2D non-uniform graph using networkx.
    Inherits from NonUniformMesh class.
    """

    def __init__(self, xy, nx, land_mask, graph_type, root_path_to_save, supplementary_masks, crossing_edeges=False):
        super().__init__(xy, nx, land_mask, graph_type, root_path_to_save, supplementary_masks)
        self.crossing_edges = crossing_edeges

    def mk_2d_non_uniform_graph(self, n_nodes_x, n_nodes_y, *args, **kwargs):
        xm, xM = np.amin(self.xy[0][0, :]), np.amax(self.xy[0][0, :])
        ym, yM = np.amin(self.xy[1][:, 0]), np.amax(self.xy[1][:, 0])

        # avoid nodes on border
        min_distance_to_border = np.sqrt(0.5)
        dx = (xM - xm) / n_nodes_x
        dy = (yM - ym) / n_nodes_y
        #lx = np.linspace(xm + min_distance_to_border, xM - min_distance_to_border, n_nodes_x, dtype=np.float32)
        #ly = np.linspace(ym + min_distance_to_border, yM - min_distance_to_border, n_nodes_y, dtype=np.float32)
        lx = np.linspace(xm + dx / 2, xM - dx / 2, n_nodes_x, dtype=np.float32)
        ly = np.linspace(ym + dy / 2, yM - dy / 2, n_nodes_y, dtype=np.float32)
        # for plotting purposes
        X, Y = np.meshgrid(lx, ly)                  # both shape (n_nodes_y, n_nodes_x)
        coords = np.column_stack((X.ravel(), Y.ravel()))
        counts = mm.count_nodes_in_masks(self.supplementary_masks, coords)
        title = f"total nodes for each mask in level {n_nodes_x}.svg"
        filepath = os.path.join(self.path_to_save_figures, self.graph_type, title)
        mm.plot_counts_table(counts, title, filepath)
        
        mg = np.meshgrid(lx, ly)
        g = networkx.grid_2d_graph(len(ly), len(lx))

        # kdtree for nearest neighbor search of land nodes
        land_points = np.argwhere(self.land_mask.T).astype(np.float32)
        land_kdtree = scipy.spatial.KDTree(land_points)

        # add nodes excluding land
        for node in list(g.nodes):
            node_pos = np.array([mg[0][node], mg[1][node]], dtype=np.float32)
            dist, _ = land_kdtree.query(node_pos, k=1)
            if dist < np.sqrt(0.5):
                g.remove_node(node)
            else:
                g.nodes[node]["pos"] = node_pos

        # add diagonal edges if both nodes exist
        if self.crossing_edges is True:
            for x in range(n_nodes_x - 1):
                for y in range(n_nodes_y - 1):
                    if g.has_node((x, y)) and g.has_node((x + 1, y + 1)):
                        g.add_edge((x, y), (x + 1, y + 1))
                    if g.has_node((x + 1, y)) and g.has_node((x, y + 1)):
                        g.add_edge((x + 1, y), (x, y + 1))
        else:
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
    
    def create_mesh_levels(self, mesh_levels, plot=False,resolutions_list=None ,  *args, **kwargs):
        if resolutions_list is not None:
            if len(resolutions_list) != mesh_levels:
                raise ValueError("Number of levels and number of resolutions must be equal")
            G_list = []
            for lev in range(1, mesh_levels + 1):
                n = resolutions_list[lev - 1]
                print(f"level {lev}, n: {n}")
                g = self.mk_2d_non_uniform_graph(n, n, *args, **kwargs)
                if plot == True:
                    mm.plot_graph(
                        from_networkx(g),
                        sea_surface=self.land_mask,
                        title=f"Mesh graph, level {lev}",
                        graph_type=self.graph_type,
                        save_dir=self.path_to_save_figures, 
                        xlim=(0, self.land_mask.shape[1]),
                        ylim=(0, self.land_mask.shape[0]),
                    )
                    g_metrics = mm.metrics_from_networkx_graph(g, None)
                    mm.plot_degree_distribution(
                        g_metrics["count_degrees"], title=f"Mesh graph level {lev} nodes degree",
                        graph_type=self.graph_type, save_dir=self.path_to_save_figures)
                    mm.create_metrics_table(
                        g_metrics, None,
                        columns_names=["nodes"],
                        metrics_to_avoid=["count_degrees"],
                        title=f"Mesh graph level {lev} metrics", graph_type=self.graph_type,
                        save_dir=self.path_to_save_figures
                        )
                G_list.append(g)
            return G_list
        return super().create_mesh_levels(mesh_levels, plot, *args, **kwargs)

    def add_down_edges(self, G_down, v_to_list, v_from_list, kdt_m, k_neighboors,*args, **kwargs):
        mc.add_edges_based_in_k_neighboors(G_down, v_from_list, v_to_list, kdt_m, self.land_mask, k_neighboors=k_neighboors)

    def establish_edges_g2m(self, graph, sender_nodes_list, receiver_nodes_list, required_connections,*args, **kwargs):
        receiver_nodes_list = [node for node in receiver_nodes_list if node[0] == 0]
        mc.establish_n_edges_g2m(graph, sender_nodes_list, receiver_nodes_list, required_connections=required_connections)

    def establish_edges_m2g(self, graph, *args, **kwargs):
        G_m2g = mc.create_reversed_graph(graph)
        return G_m2g
    


        