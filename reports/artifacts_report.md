# Description of results
---

The analysis of the RMSE gradient shows that the spatial structure of the error is organized in tiles defined by the Grid–Mesh association. When connectivity is minimal ((k = 1)), the partition is almost equivalent to a first-order Voronoi diagram. Each mesh node (magenta points) defines a convex region, and the edges between tiles correspond to equal-distance boundaries. In this configuration, the geometry is simple and the discontinuities are clearly aligned with proximity-based separation.

When (k) increases, the geometry of the tiles changes significantly. The regions no longer depend on a single dominant node but on overlapping neighborhoods. This produces irregular and non-convex shapes, with some zones showing fragmentation or elongated polygons. The increase in geometric complexity is not spatially uniform: certain areas retain simple tiles, while others show strong subdivision.

![alt text](./figures/artifact_uniform_k_connections.png)

The RMSE gradient highlights the tile borders. These borders consistently concentrate error discontinuities, regardless of connectivity level. However, the orientation of the boundaries suggests that some of them align with dominant physical structures in the domain, which introduces anisotropy in how the error accumulates.

In configurations with higher connectivity, vertices where three or more tiles converge are more frequent. These points show locally intensified gradients and are likely to act as sources or transition nodes for error growth under autoregressive prediction.

Higher connectivity does not imply uniform improvement. Although the average tile size decreases with increasing (k), the irregularity and fragmentation of the geometry increase spatial heterogeneity in the error. In several areas, increasing connectivity reinforces boundary complexity instead of reducing discontinuities.

Overall, the results indicate that the spatial structure of the RMSE is determined by the geometry produced by the Grid–Mesh association. Both connectivity ((k)) and node placement control the location and shape of the discontinuities, as well as the zones where error gradients accumulate.


Perfecto. Voy a redactarlo como una **sección continua de informe técnico**, iniciando con el contexto de lo realizado, seguido del análisis y cierre conclusivo. No incluiré encabezados ni subdivisiones explícitas, manteniendo un flujo técnico-descriptivo claro.

## Order-k Voronoi Diagrams
---

The analysis focused on determining the geometric origin of the spatial artifacts observed in the squared error fields of the prediction system. These artifacts consistently appeared as contiguous tiles with clearly defined boundaries, suggesting a structural rather than stochastic cause. To investigate the source of this behavior, the connectivity mechanism used between the grid nodes and the mesh nodes was examined. This mechanism relies on a k-nearest neighbors (k-NN) search implemented through a KD-Tree, where each grid node is associated with its k closest mesh nodes based on Euclidean distance. The hypothesis was that this association rule induces a spatial partition analogous to an order-k Voronoi diagram.

To validate this, a controlled experiment was conducted by replicating the theoretical construction of order-k Voronoi partitions described in “Order-k Voronoi Diagrams, k-Sections, and k-Sets.” Using the same k-NN approach applied in the prediction framework, a synthetic mesh of nodes was placed over a 300×300 domain, and the domain was partitioned by selecting the k closest nodes to each query point. The resulting tessellation reproduced the exact structure described in the reference work, confirming that the connectivity mechanism is geometrically equivalent to an order-k Voronoi partition.

![alt text](./figures/k_voronoi_fig_paper.png)

After this verification, the same procedure was used to approximate the tile patterns visually identified in the artifact regions for the uniform configuration at k = 5. The replicated partition matched the artifact geometry, validating that the tiling observed in the gradient of the RMSE is a direct consequence of the k-NN connectivity rule. The grid–mesh association does not generate arbitrary subdivisions; it imposes a deterministic partition of space where each region corresponds to points sharing the same set of k nearest mesh nodes.

![alt text](./figures/k_5_voronoi_uniform_config.png)

The resulting tiles are piecewise-defined regions, and each boundary marks a discontinuity in the neighbor set. These boundaries coincide with zones of elevated gradient in the squared error fields, indicating that the error discontinuities are not numerical noise but the direct expression of the underlying partition geometry. As k increases, the partitions become more complex and less convex, and additional intersections appear where three or more tiles meet. These vertices act as localized pivots where the ranking of neighbors changes more abruptly. In autoregressive prediction schemes, each tile behaves as a semi-independent predictor whose error develops its own trajectory, and the discontinuities at the boundaries accumulate over time.

Domain geometry further modulates these structures. Coastal boundaries truncate tiles and introduce anisotropy by removing potential neighbor candidates, producing elongated or curved subdivisions in some areas. Additionally, the use of Euclidean metric in latitude–longitude space introduces directional distortion, particularly in east–west orientation, which biases the shape of the cells and the alignment of the boundaries.

The analysis also shows that increasing connectivity alone does not reduce the artifacts. Higher k reduces the average area of the tiles but increases their irregularity and fragmentation, leading to stronger heterogeneity in the spatial distribution of the error. The discontinuities persist because the fundamental mechanism remains discrete: the transition between neighbor sets is abrupt, not gradual.

From these observations, it is concluded that the origin of the artifacts lies in the geometric properties of the k-NN-based association, which is mathematically equivalent to constructing an order-k Voronoi diagram over the set of mesh nodes. This equivalence explains both the persistence of the discontinuities and their spatial organization. The error accumulates according to the partitioning, not uniformly across the domain, and the geometry of the tiles dictates where gradients are concentrated.

These findings indicate that mitigation strategies must act on the geometry of the partition rather than on global smoothing or increasing the number of neighbors. Potential adjustments include softening the connectivity rule to reduce hard boundaries, redefining the metric to better align with domain characteristics, redistributing nodes to regularize tile shapes, or introducing weighted or adaptive neighbor selection. Without modifying the structure that induces the order-k tessellation, the artifacts will remain intrinsic to the prediction system.

---

Claro. Aquí tienes una versión integrada que combina ambas interpretaciones en un solo análisis coherente y unificado, sin contradicciones y con énfasis en la interacción entre geometría, conectividad y ubicación costera:

## Error analysis by tile
---

The spatial artifacts observed in the RMSE fields originate from the connectivity mechanism used to associate grid nodes with mesh nodes via a k-nearest neighbors (k-NN) search. This rule induces a deterministic order-(k) Voronoi-type partition of the domain, where each tessellation cell contains all points that share the same set of (k) nearest mesh nodes. The discontinuities in the error fields align with the boundaries between these regions, since a change in the neighbor set translates into a discrete change in the local predictive model.

However, the geometry of the tessellation cannot be interpreted solely from the k-order Voronoi construction in an unbounded or homogeneous domain. The coastline acts as an active geometric constraint that truncates and distorts the cells. Nearshore regions systematically produce smaller, asymmetric, and irregular tessellation shapes because part of the potential spatial domain is unavailable, limiting the effective area that any mesh node can represent. Offshore regions, in contrast, preserve more canonical partition shapes that reflect the underlying k-NN structure with less distortion.

This geometric asymmetry is coupled with the physical structure of the field being predicted. Coastal areas tend to exhibit steeper environmental gradients and higher spatial variability. As a result, cells located near the coast not only inherit a more constrained geometry but also operate over regions where the predictive task is intrinsically harder. This dual effect explains why the histograms of RMSE within coastal tessellation cells display skewed or heavy-tailed distributions, while offshore cells tend to show more symmetric, near-Gaussian behavior.

![alt text](./figures/artifacts_uniform_config_1_k_connection.png)
![alt text](./figures/error_hist_by_tile_1_k_connec.png)

Temporal evolution of the per-cell RMSE further confirms this coupling. When the mean RMSE is computed per tessellation across lead times, the growth curves are approximately linear, but their slopes vary widely. The steepest slopes correspond to the tessellation cells located near the coastline, where the combination of geometric truncation and physical heterogeneity leads to faster error accumulation. Offshore and interior cells, by contrast, show lower slopes and more homogeneous temporal behavior.

![alt text](./figures/uniform_conf_rmse_curve_1_k_connec_by_leadtime.png)
![alt text](./figures/rmse_slope_uniform_config_k_1_connec_by_tile.png)

These findings indicate that the spatial heterogeneity in both instantaneous error distribution and temporal error growth is not merely a byproduct of the partitioning rule itself, but emerges from the interaction between the order-(k) Voronoi geometry and the coastline boundary. The connectivity rule defines the baseline partition, but the boundary conditions and underlying physical gradients determine how each cell’s geometry, error distribution, and slope of RMSE growth diverge from one another.

In short, the tessellation structure and its error dynamics cannot be decoupled from geographic position. Coastal proximity simultaneously increases geometric distortion, reduces coverage area, and amplifies the physical gradients that drive prediction error. Offshore cells remain closer to the idealized k-order Voronoi geometry and accumulate error more slowly. The result is a spatially organized stratification of error behavior that reflects both the mathematical structure of the connectivity mechanism and the physical constraints of the domain.

