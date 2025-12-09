#abstract class for non-uniform meshes
from abc import ABC, abstractmethod
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx
import os
import skimage.draw
import networkx
import scipy.spatial
from typing import List, Tuple, Dict, Any

import mesh_models.mesh_metrics as mm


class NonUniformMesh(ABC):
    """
    Abstract base class for non-uniform meshes.
    """

    def __init__(self, xy, nx, land_mask, graph_type, root_path_to_save, supplementary_masks=None):
        """
        Initialize the NonUniformMesh object.
        :param xy: Coordinates of the mesh points, specifying the grid dimensions.
        :param nx: number of children (nx**2) for the mesh levels.
        :param land_mask: Land mask array indicating land and ocean points.
        :param graph_type: Type of graph to be created: hierarchical or combined.
        Note: root path to save should have the hierarchical or combined folder in it, 
        graph_type is recommended to be used to specify the type of mesh to be created.
        :param root_path_to_save: Root path to save the generated graphs and figures.
        :param supplementary_masks: Dictionary of supplementary masks for additional features.
        """
        self.nx = nx
        self.land_mask = land_mask
        self.xy = xy
        self.pos_max = torch.max(torch.abs(
            torch.tensor(xy))
            )
        self.graph_type = graph_type
        self.root_path_to_save = root_path_to_save
        self.path_to_save_figures = os.path.join(
            self.root_path_to_save, "figures", self.graph_type
        )
        self.path_to_save_graphs = os.path.join(
            self.root_path_to_save, "graphs", self.graph_type
        )
        os.makedirs(self.path_to_save_figures, exist_ok=True)
        os.makedirs(self.path_to_save_graphs, exist_ok=True)
        self.supplementary_masks = supplementary_masks

    def sort_nodes_internally(self, nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
        H = networkx.DiGraph()
        H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
        H.add_edges_from(nx_graph.edges(data=True))
        return H

    def crosses_land(self, node1, node2, threshold=8):
        x1, y1 = node1
        x2, y2 = node2
        rr, cc = skimage.draw.line(round(y1), round(x1), round(y2), round(x2))
        return np.sum(self.land_mask[rr, cc]) >= threshold

    def from_networkx_with_start_index(self, nx_graph, start_index):
        pyg_graph = from_networkx(nx_graph)
        pyg_graph.edge_index += start_index
        return pyg_graph
    
    def save_edges(self, graph, name):
        torch.save(
            graph.edge_index, os.path.join(self.path_to_save_graphs, f"{name}_edge_index.pt")
        )
        edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
            torch.float32
        )  # Save as float32
        torch.save(edge_features, os.path.join(self.path_to_save_graphs, f"{name}_features.pt"))

    def prepend_node_index(self,graph, new_index):
        # Relabel node indices in graph, insert (graph_level, i, j)
        ijk = [tuple((new_index,) + x) for x in graph.nodes]
        to_mapping = dict(zip(graph.nodes, ijk))
        return networkx.relabel_nodes(graph, to_mapping, copy=True)
    
    def save_edges_list(self, graphs, name):
        torch.save(
            [graph.edge_index for graph in graphs],
            os.path.join(self.path_to_save_graphs, f"{name}_edge_index.pt"),
        )
        edge_features = [
            torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
                torch.float32
            )
            for graph in graphs
        ]  # Save as float32
        torch.save(edge_features, os.path.join(self.path_to_save_graphs, f"{name}_features.pt"))


    def save_metrics(self, from_level, to_level, networkx_graph, pyg_graph, common_title, save_dir):
        mm.plot_graph(
                    pyg_graph,
                    title=f"{common_title} graph",
                    graph_type=self.graph_type, save_dir=save_dir
                )
        mm.plot_graph_networkx(
                    networkx_graph, title=f"{common_title} sender layer",level=from_level, graph_type=self.graph_type,
                    save_dir=save_dir
                )
        mm.plot_graph_networkx(
                    networkx_graph, title=f"{common_title} receiver layer",level=to_level, graph_type=self.graph_type,
                    save_dir=save_dir
                )
        g_from_metrics = mm.metrics_from_networkx_graph(networkx_graph, from_level)
        
        g_to_metrics = mm.metrics_from_networkx_graph(networkx_graph, to_level)
        mm.plot_degree_distribution(
                    g_from_metrics["count_degrees"], title=f"{common_title} sender layer nodes degree", 
                    graph_type=self.graph_type, save_dir=save_dir
                )
        mm.plot_degree_distribution(
                    g_to_metrics["count_degrees"], title=f"{common_title} receiver layer nodes degree",
                    graph_type=self.graph_type, save_dir=save_dir
                )
        mm.create_metrics_table(
                    g_from_metrics, g_to_metrics,
                    columns_names=["sender_nodes", "receiver_nodes"],
                    metrics_to_avoid=["count_degrees"],
                    title=f"{common_title} metrics", graph_type=self.graph_type,
                    save_dir=save_dir
                )  
    @abstractmethod
    def mk_2d_non_uniform_graph(self, n_nodes_x, n_nodes_y, *args, **kwargs):
        """"
        Create a 2D non-uniform graph.
        This metod should be implemented in subclasses where the graph is created based on the strategy to use.

        :param n_nodes_x: Number of nodes in the x direction.
        :param n_nodes_y: Number of nodes in the y direction.
        :param args: Additional arguments for graph creation.
        :param kwargs: Additional keyword arguments for graph creation.

        """
        pass

    def create_mesh_levels(self, mesh_levels, plot=False, *args, **kwargs):
        """
        Create mesh levels for the non-uniform mesh.
        :param levels: List of levels for the mesh.
        :param plot: Boolean indicating whether to plot the graph.

        returns:
        - G_list: List of grid objects for each level.
        """
        nlev = int(np.log(max(self.xy.shape)) / np.log(self.nx))
        nleaf = self.nx**nlev  # leaves at the bottom = nleaf**2
        print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")
        # multi resolution tree levels
        G_list = []
        for lev in range(1, mesh_levels + 1):
            n = int(nleaf / (self.nx**lev))
            print(f"level {lev}, n: {n}")
            g = self.mk_2d_non_uniform_graph(n, n, *args, **kwargs)
            if plot == True:
                mm.plot_graph(
                    from_networkx(g),
                    title=f"Mesh graph, level {lev}",
                    sea_surface=self.land_mask,
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
    
    def prepare_hierarchical_graphs(self, graph_list):
        """
        Preparation step where we add the level to each node in the graph
        and extract the first node index in each level.
        
        """
        graph_list = [
            self.prepend_node_index(graph, level_i)
            for level_i, graph in enumerate(graph_list)
        ]

        num_nodes_level = np.array([len(g_level.nodes) for g_level in graph_list])
        # First node index in each level in the hierarchical graph
        first_index_level = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
        )
        return graph_list, first_index_level
    
    @abstractmethod
    def add_down_edges(self, G_down, v_to_list, v_from_list, *args, **kwargs):
        """
        method to add down edges to a lower resolution mesh level.
        :param G_down: Graph object for the lower resolution mesh level.
        :param v_to_list: List of nodes in the lower resolution mesh level.
        :param v_from_list: List of nodes in the higher resolution mesh level.
        :param
        :return: None
        """
        pass

    def create_hierarchical_mesh_up_mesh_down_edges(self, graph_list, mesh_levels, first_index_level, plot=False, *args, **kwargs):
        up_graphs = []
        down_graphs = []
        for from_level, to_level, G_from, G_to, start_index in zip(
            range(1, mesh_levels),
            range(0, mesh_levels - 1),
            graph_list[1:],
            graph_list[:-1],
            first_index_level[: mesh_levels - 1],
        ):
            # start out from graph at from level
            G_down = G_from.copy()
            G_down.clear_edges()
            G_down = networkx.DiGraph(G_down)

            # Add nodes of to level
            G_down.add_nodes_from(G_to.nodes(data=True))

            # build kd tree for mesh point pos
            # order in vm should be same as in vm_xy
            v_to_list = list(G_to.nodes)
            v_from_list = list(G_from.nodes)
            v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
            kdt_m = scipy.spatial.KDTree(v_from_xy)

            # add edges from mesh to grid
            self.add_down_edges(G_down, v_to_list, v_from_list,kdt_m, *args, **kwargs)

            # relabel nodes to integers (sorted)
            G_down_int = networkx.convert_node_labels_to_integers(
                G_down, first_label=start_index, ordering="sorted"
            )  # Issue with sorting here
            G_down_int = self.sort_nodes_internally(G_down_int)
            pyg_down = self.from_networkx_with_start_index(G_down_int, start_index)

            # Create up graph, invert downwards edges
            up_edges = torch.stack(
                (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
            )
            pyg_up = pyg_down.clone()
            pyg_up.edge_index = up_edges

            up_graphs.append(pyg_up)
            down_graphs.append(pyg_down)

            if plot:
                common_title = f"Mesh down {from_level} to {to_level}"
                save_dir = os.path.join(self.path_to_save_figures, "mesh_down")
                os.makedirs(save_dir, exist_ok=True)
                self.save_metrics(from_level, to_level, G_down, pyg_down, common_title=common_title, save_dir=save_dir)
                
                common_title = f"Mesh up {to_level} to {from_level}"
                save_dir = os.path.join(self.path_to_save_figures, "mesh_up")
                os.makedirs(save_dir, exist_ok=True)
                self.save_metrics(to_level, from_level, G_down, pyg_down, common_title=common_title, save_dir=save_dir)

        # Save up and down edges
        self.save_edges_list(up_graphs, "mesh_up")
        self.save_edges_list(down_graphs, "mesh_down") 
 

    def create_combined_m2m_graph(self, graph_list, plot = False):
        """
        Mesh-to-mesh graph for the non-uniform mesh for the combined type graph
        Note: This is a version of a mesh structure that in the moment is not working properly 
        and is not currently used in the code. This function is kept for future reference and as a placeholder
        """
        
        G_tot = graph_list[0]
        for lev in range(1, len(graph_list)):
            nodes = list(graph_list[lev - 1].nodes)
            n = int(np.sqrt(len(nodes)))
            ij = (
                np.array(nodes)
                .reshape((n, n, 2))[1::self.nx, 1::self.nx, :]
                .reshape(int(n / self.nx) ** 2, 2)
            )
            ij = [tuple(x) for x in ij]
            graph_list[lev] = networkx.relabel_nodes(graph_list[lev], dict(zip(graph_list[lev].nodes, ij)))
            G_tot = networkx.compose(G_tot, graph_list[lev])

        # Relabel mesh nodes to start with 0
        G_tot = self.prepend_node_index(G_tot, 0)

        # relabel nodes to integers (sorted)
        G_int = networkx.convert_node_labels_to_integers(
            G_tot, first_label=0, ordering="sorted"
        )

        # Graph to use in g2m and m2g
        #G_bottom_mesh = G_tot
        #all_mesh_nodes = G_tot.nodes(data=True)

        # export the nx graph to PyTorch geometric
        pyg_m2m = from_networkx(G_int)
        m2m_graphs = [pyg_m2m]
        mesh_pos = [pyg_m2m.pos.to(torch.float32)]

        if plot:
            mm.plot_graph(pyg_m2m, title="Mesh-to-mesh", graph_type=self.graph_type, save_dir=self.path_to_save_figures)

        self.save_edges_list(m2m_graphs, "m2m")

        # Divide mesh node pos by max coordinate of grid cell
        mesh_pos = [pos / self.pos_max for pos in mesh_pos]

        # Save mesh positions
        torch.save(
            mesh_pos, os.path.join(self.path_to_save_graphs, "mesh_features.pt")
        )  # mesh pos, in float32

    def create_hierarchical_m2m_graph(self, graph_list, first_index_level):
        """
        Create a mesh-to-mesh graph for the non-uniform mesh where we save the edges in PyG format.
        :param graph_list: List of graphs for each level.
        :param first_index_level: First node index in each level in the hierarchical graph.
        :return: List of mesh-to-mesh graphs.
        """
        m2m_graphs = [
            self.from_networkx_with_start_index(
                networkx.convert_node_labels_to_integers(
                    level_graph, first_label=start_index, ordering="sorted"
                ),
                start_index,
            )
            for level_graph, start_index in zip(graph_list, first_index_level)
        ]

        self.save_edges_list(m2m_graphs, "m2m")
        return m2m_graphs

    def create_mesh_features(self, m2m_graphs):
        """
        Create mesh features for the non-uniform mesh where the features are the mesh positions 
        normalized by the maximum position.
        :param m2m_graphs: List of mesh-to-mesh graphs.
        :return: None 
        """
        mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]
        mesh_pos = [pos / self.pos_max for pos in mesh_pos]

    # Save mesh positions
        torch.save(
            mesh_pos, os.path.join(self.path_to_save_graphs, "mesh_features.pt")
        )  # mesh pos, in float32
    
    
    def create_g2m_graph(self, graph_list, plot=False, *args, **kwargs):
        """
        Create G2M graph for the non-uniform mesh where we save the edges in PyG format.
        :param graph_list: List of graphs for each level.
        """
        G_bottom_mesh = graph_list[0]

        joint_mesh_graph = networkx.union_all([graph for graph in graph_list])
        all_mesh_nodes = joint_mesh_graph.nodes(data=True)

        # mesh nodes on lowest level
        G_grid, vm_xy, vm = self.create_grid_graph(G_bottom_mesh)
        vg_list = list(G_grid.nodes)
        G_grid.add_nodes_from(all_mesh_nodes)

        G_g2m = networkx.Graph()
        G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

        # turn into directed graph
        G_g2m = networkx.DiGraph(G_g2m)


        self.establish_edges_g2m(
            G_g2m, vg_list, vm, *args, **kwargs
        )
        
        pyg_g2m = from_networkx(G_g2m)

        if plot:
            pyg_g2m_reversed = pyg_g2m.clone()
            pyg_g2m_reversed.edge_index = pyg_g2m.edge_index[[1, 0]]
            common_title = "Grid-to-mesh"
            from_level, to_level = 1000, 0
            save_dir = os.path.join(self.path_to_save_figures, "G2M")
            os.makedirs(save_dir, exist_ok=True)
            self.save_metrics(from_level, to_level, G_g2m, pyg_g2m_reversed, common_title=common_title, save_dir=save_dir)
            
        self.save_edges(pyg_g2m, "g2m")
            
        return G_g2m


    def create_grid_graph(self, G_bottom_mesh):
        vm = G_bottom_mesh.nodes
        vm_xy = np.array([xy for _, xy in vm.data("pos")])

        Ny, Nx = self.xy.shape[1:]

        G_grid = networkx.grid_2d_graph(Ny, Nx)
        G_grid.clear_edges()

        # vg features (only pos introduced here)
        nodes_to_remove = []
        for node in G_grid.nodes:
            # Remove the node from the graph if it is a land node
            if self.land_mask[node[0], node[1]]:
                nodes_to_remove.append(node)
            else:
                # pos is in feature but here explicit for convenience
                G_grid.nodes[node]["pos"] = np.array([self.xy[0][node], self.xy[1][node]])

        for node in nodes_to_remove:
            G_grid.remove_node(node)

        G_grid = self.prepend_node_index(G_grid, 1000)
        
        return G_grid, vm_xy, vm
    
    @abstractmethod
    def establish_edges_g2m(self, graph, sender_nodes_list, receiver_nodes_list, *args, **kwargs):
        """
        Establish edges for the G2M graph.
        :param graph: Graph object for the G2M graph.
        :param sender_nodes_list: List of nodes in the sender graph.
        :param receiver_nodes_list: List of nodes in the receiver graph.
        :param
        :return: None
        """
        pass

    def create_m2g_graph(self, G_g2m, plot=False, *args, **kwargs):
        G_m2g = G_g2m.copy()

        G_m2g = self.establish_edges_m2g(G_g2m, *args, **kwargs)
        G_m2g_int = networkx.convert_node_labels_to_integers(
            G_m2g, first_label=0, ordering="sorted"
        )
        pyg_m2g = from_networkx(G_m2g_int)

        if plot:
            common_title = "Mesh-to-Grid"
            from_level, to_level = 0, 1000
            save_dir = os.path.join(self.path_to_save_figures, "m2g")
            os.makedirs(save_dir, exist_ok=True)
            self.save_metrics(from_level, to_level, G_m2g, pyg_m2g, common_title=common_title, save_dir = save_dir)

        self.save_edges(pyg_m2g, "m2g")

    @abstractmethod
    def establish_edges_m2g(self, graph,*args, **kwargs):
        """
        Establish edges for the G2M graph.
        :param graph: Graph object for the G2M graph.
        :param args: Additional arguments for edge establishment.
        :param kwargs: Additional keyword arguments for edge establishment.
        :return: graph: Graph object with established edges.
        """
        pass