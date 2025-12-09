# **Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware Meshes in GNNs for Sea Surface Temperature Forecasting**

> Official repository for the paper: *Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware Meshes in Graph Neural Networks for Sea Surface Temperature Forecasting* (2025).

-----

## ✨ What’s in this repo

  * **End-to-end pipelines:** Download & preprocess Copernicus SST, ERA5 Winds, and ETOPO Bathymetry.
  * **Modular Graph Construction:**
      * Grid→Mesh coupling via **k-NN**.
      * Four mesh families: **UC** (Crossed), **U** (Uniform), **B** (Bathymetry-weighted), **F** (FPS-Adaptive).
  * **Forecasting Model:** A lightweight **Encoder–Processor–Decoder GNN** (Interaction Network).
  * **Reproducible Experiments:**
      * **Exp 1 (Connectivity):** Effect of coupling density ($k=1 \dots 5$).
      * **Exp 2 (Density):** Node density sweep using geometric progression ($\varphi \approx 1.618$).
  * **Diagnostic Tools:**
      * **Proof of Concept:** Code demonstrating the algebraic equivalence between k-NN coupling and **Order-k Voronoi diagrams**.
      * **Spatial Analysis:** "Error Analysis by Spatial Tessellation" (per-tile RMSE growth).

-----

## 🔍 Problem & Key Contributions (TL;DR)

  * **The Problem:** Standard grid→mesh coupling in GNNs uses k-NN links whose geometry is typically treated as a black box. We mathematically prove and empirically show that this coupling **induces Order-k Voronoi partitions** that act as semi-isolated predictors, seeding **polygonal seams** and driving structured error growth.
  * **Key Findings:**
    1.  **Geometric Determinism:** Forecast errors align perfectly with **Order-k Voronoi boundaries** implied by the encoder/decoder ($k$-NN).
    2.  **Bathymetry-Awareness:** Unstructured meshes (**B-mesh, F-mesh**) with **k=3–4** break these artifacts, improving long-range coherence and reducing RMSE by **up to 30%** vs. structured baselines.
    3.  **The "More Nodes" Fallacy:** Simply adding nodes to structured meshes (U/UC) **does not** improve accuracy and can degrade performance due to edge redundancy. Unstructured meshes only diverge statistically (Bayesian confidence) after a critical resolution threshold.
  * **Takeaway:** Treat coupling as a **first-class design knob**. Use **k≈3–4** and distribute latent capacity based on **coastal gradients** ($D \propto 1/\sqrt{B}$) rather than uniform grids.

-----

## 📦 Repository Layout

```
grid2mesh-voronoi-artifacts/
├───docs
├───neural_lam
│   └───models
├───notebooks
│   ├───article_analysis
│   └───article_review
├───reports
│   └───figures
├───seacast_cli
│   ├───bin
│   ├───config
│   └───lib
├───src
│   └───seacast_tools
│       └───mesh_models
│           └───supplementary_masks
└───tests
    ├───e2e
    ├───integration
    └───unit
```

-----

## 📥 Data

We use **public** datasets covering the Canary Islands & NW Africa ($19.55^{\circ}N$ to $34.52^{\circ}N$, $-20.97^{\circ}E$ to $-5.98^{\circ}E$).

  * **SST:** Copernicus Marine `SST_reanalysis` (Daily, 0.05°).
  * **Winds:** ECMWF ERA5 (u/v components, 10m).
  * **Bathymetry:** NOAA ETOPO Global Relief Model.

### Download & Preprocess
The **`download_data`** script is a wrapper to simplify data acquisition from CMEMS.

**Instructions:**

1.  **Open the `download_data` script.**
2.  **Enter your CMEMS `user` and `password`** manually into the variables provided at the beginning of the file.
3.  **Make the script executable:**
    ```bash
    chmod +x download_data
    ```
4.  **Run the script:**
    ```bash
    ./download_data
    ```
-----

## 🧱 Mesh Families & Mathematical Formulation

We implement four mesh generation strategies with consistent node counts per level:

1.  **UC-mesh (Uniform Crossed):** Structured grid with diagonals (8-connectivity). High redundancy.
2.  **U-mesh (Uniform):** Structured grid, Delaunay triangulation (no crossing edges).
3.  **B-mesh (Bathymetry-Aware):** Nodes sampled via probability distribution inverse to depth.
4.  **F-mesh (FPS + Balanced):** Uses **Farthest Point Sampling** on a mixed-sigmoid distribution to balance coastal detail with open-ocean coverage.

**Coupling:** Connect observational grid $v^G$ to mesh $v^M$ using **k-NN**.

  * *Theory:* This induces partitions $\text{VP}_i$ defined by unique generator subsets $P^k_i$, mathematically equivalent to **Order-k Voronoi diagrams**.

-----

## 🚀 Quickstart

**Train F-mesh (k=4) at High Resolution (159 nodes):**

```bash
python -m src.train \
  --data_dir data/processed \
  --mesh meshes/cache/F_159_78.pkl \
  --k 4 \
  --config configs/train.yaml \
  --out runs/fmesh_k4_159_78
```

**Evaluate & Diagnose Artifacts:**

```bash
# Generate RMSE maps, Gradients, and Voronoi Overlays
python -m src.viz.make_figures \
  --run_dir runs/fmesh_k4_159_78 \
  --plot_type "voronoi_diagnostic"
```

-----

## 🔁 Reproducing Paper Experiments

### Experiment 1: Connectivity Sweep ($k=1 \dots 5$)

Tests the hypothesis that intermediate $k$ smooths Voronoi artifacts.

```bash
bash experiments/exp1_connectivity/run_all.sh --meshes U UC B F --k_list "1 2 3 4 5"
```

  * **Result:** $k=3$ (B-mesh) and $k=4$ (F-mesh) yield lowest RMSE. $k=1$ produces sharp tessellation discontinuities.

### Experiment 2: Node Density Sweep

Increases nodes by Golden Ratio $\varphi \approx 1.618$ (Levels: 14, 20, 34, 52, 159).

```bash
bash experiments/exp2_density/run_all.sh --meshes U UC B F --scale_factor 1.618
```

  * **Result:** Statistical divergence (Bayesian Intervals) appears only after Day 6. Unstructured meshes outperform structured ones significantly at high resolutions; structured meshes degrade due to "edge redundancy."

-----

## 📊 Expected Results (Sanity Checks)

1.  **Artifact Maps:** For $k=1$, $|\nabla \text{RMSE}|$ maps should perfectly match analytical Voronoi boundaries.
2.  **Performance Band:** **B-mesh (k=3)** and **F-mesh (k=4)** should be the top performers ($\sim$0.24 K RMSE).
3.  **Regime Imbalance:** RMSE should be significantly higher in coastal waters ($<500m$) due to data scarcity and high variability.
4.  **Cost:** Training time increases linearly ($\sim$30% per added $k$).

-----

## 🛠️ Environment & Dependencies

  * **Python:** 3.10
  * **Core:** PyTorch, PyTorch Geometric (or scatter/sparse)
  * **Data:** Xarray, NetCDF4, NumPy
  * **Geo/Viz:** Cartopy, Shapely, Matplotlib
  * **Optional:** `pykeops` (for fast k-NN on large grids), `potpourri3d` (for FPS).

-----

## 🧾 Citation

```bibtex
@article{Cuervo2025Voronoi,
  title={Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware Meshes in Graph Neural Networks for Sea Surface Temperature Forecasting},
  author={Cuervo-Londoño, Giovanny A. and Reyes, José G. and Rodríguez-Santana, Ángel and Sánchez, Javier},
  journal={Electronics},
  year={2025},
  publisher={MDPI}
}
```

-----

## 🙌 Acknowledgments

Supported by **ECOAQUA** and **CTIM** (ULPGC). Data provided by **Copernicus Marine Service**, **ECMWF**, and **NOAA**.

**Corresponding Author:** Giovanny A. Cuervo-Londoño (`giovanny.cuervo101@alu.ulpgc.es`)
