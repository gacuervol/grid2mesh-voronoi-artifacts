# **Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware Meshes in GNNs for Sea Surface Temperature Forecasting**

> Reproducible code and assets for the paper: *Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware Meshes in Graph Neural Networks for Sea Surface Temperature Forecasting*.

---

## ✨ What’s in this repo

* End-to-end pipelines to **download & preprocess** SST, winds and bathymetry.
* Modular **graph construction**: grid→mesh coupling, four mesh families, and k-NN enc/dec links.
* A lightweight **encoder–processor–decoder GNN** for SST forecasting.
* Exact scripts to reproduce:

  * **Experiment 1**: effect of grid→mesh connectivity (**k=1…5**).
  * **Experiment 2**: effect of **node density** (five resolutions).
* **Artifact diagnostics**: order-k Voronoi tilings, per-tile RMSE growth, gradient maps.
* Plotting utilities to regenerate all figures shown in the paper.

---

## 🔍 Problem & Key Contributions (TL;DR)

* **Problem.** Standard grid→mesh coupling in GNN forecasters uses k-NN links whose geometry is usually ignored. We show this coupling **induces order-k Voronoi partitions** that **seed polygonal seams** and drive error growth at long horizons.
* **Findings.**

  1. Errors align with **order-k Voronoi boundaries** implied by the encoder/decoder k-NN.
  2. **Bathymetry-aware meshes** with **k=3–4** **reduce polygonal artifacts** and improve long-range coherence, reaching **up to 30% lower RMSE** vs. structured baselines in our domain.
  3. Simply adding nodes/edges is not enough; **coupling geometry** governs accuracy.
* **Takeaway.** Treat grid→mesh coupling as a **first-class design knob**; adapt node placement and pick **k≈3–4**; focus latent capacity on **coasts and steep-gradient regions** where **finer-scale structures** matter.

---

## 📦 Repository layout

```
gnn-voronoi-sst/
├─ data/
│  ├─ raw/              # NetCDF downloads (SST, winds, bathymetry)
│  └─ processed/        # Preprocessed tensors & masks
├─ meshes/
│  ├─ configs/          # JSON/YAML mesh configs (UC, U, B, F; sizes)
│  └─ cache/            # Saved mesh graphs (edge lists, coords)
├─ src/
│  ├─ data/             # download_*.py, preprocess_*.py, mask/bathymetry utils
│  ├─ graphs/           # mesh builders, k-NN coupling, Voronoi utils
│  ├─ models/           # encoder/processor/decoder GNN
│  ├─ train.py          # training loop (AdamW, cosine LR, WMSE)
│  ├─ eval.py           # RMSE by leadtime, maps, per-tile diagnostics
│  └─ viz/              # figure scripts (rmse curves, maps, tessellations)
├─ experiments/
│  ├─ exp1_connectivity/  # k = 1..5 configs & runners
│  └─ exp2_density/       # node-density sweep configs & runners
├─ configs/
│  ├─ dataset.yaml
│  ├─ model.yaml
│  ├─ train.yaml
│  └─ eval.yaml
├─ environment.yml
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## 📥 Data

We use **public** datasets; download scripts are provided.

* **SST (Copernicus Marine)** – daily reanalysis, 0.05°
  Product name: `SST_reanalysis`
* **10 m winds (ECMWF ERA5)** – u/v components
* **Bathymetry (NOAA ETOPO)** – global relief model

### Download & preprocess

```bash
# 1) Create environment
conda env create -f environment.yml && conda activate gnn-voronoi-sst

# 2) Download data (SST, winds, ETOPO)
python -m src.data.download_all --out data/raw --years 2000-2020

# 3) Preprocess: crop Canary–NW Africa box, land–sea mask, align grids
python -m src.data.preprocess \
  --in data/raw --out data/processed \
  --bbox "lat_min=19.55,lat_max=34.52,lon_min=-20.97,lon_max=-5.98" \
  --grid_res 0.05
```

The processed folder contains tensors shaped as `[time, lat, lon, features]`, masks, and standardized stats.

---

## 🧱 Mesh families & grid→mesh coupling

We provide four mesh builders (same node counts per level unless stated):

* **UC-mesh**: structured, **crossing diagonals** (higher edge count).
* **U-mesh**: structured, **Delaunay** (no crossing edges).
* **B-mesh**: **bathymetry-aware** sampling (denser near coasts via ( D(x,y)\propto 1/\sqrt{\tilde B} )).
* **F-mesh**: **balanced bathymetry** via mixed-sigmoid + **farthest-point sampling**.

Connect the observational grid to the mesh with **k-NN** links in the **encoder/decoder**; choose `k ∈ {1,…,5}`.

Generate & cache meshes:

```bash
# Structured (U, UC)
python -m src.graphs.build_mesh --type U  --n_level1 159 --n_level2 78 --out meshes/cache/U_159_78.pkl
python -m src.graphs.build_mesh --type UC --n_level1 159 --n_level2 78 --out meshes/cache/UC_159_78.pkl

# Bathymetry-aware (B, F)
python -m src.graphs.build_mesh --type B  --bathymetry data/processed/bathy.nc --out meshes/cache/B_159_78.pkl
python -m src.graphs.build_mesh --type F  --bathymetry data/processed/bathy.nc --out meshes/cache/F_159_78.pkl
```

---

## 🧠 Model

Encoder–processor–decoder **GNN** (interaction network style).

* **Loss**: **WMSE** with cosine latitude weights and per-feature std.
* **Optimizer**: AdamW (β₁=0.9, β₂=0.95), cosine LR, initial LR=1e-3.
* **Training**: default **150 epochs**, identical across runs.

---

## 🚀 Quickstart (single run)

Train **F-mesh, k=4** at 159/78 nodes:

```bash
python -m src.train \
  --data_dir data/processed \
  --mesh meshes/cache/F_159_78.pkl \
  --k 4 \
  --config configs/{dataset,model,train}.yaml \
  --out runs/fmesh_k4_159_78
```

Evaluate & plot:

```bash
python -m src.eval \
  --run_dir runs/fmesh_k4_159_78 \
  --lead_max 15 \
  --out runs/fmesh_k4_159_78/eval

# Figures: RMSE curves, maps, Voronoi overlays, per-tile RMSE growth
python -m src.viz.make_figures --run_dir runs/fmesh_k4_159_78
```

---

## 🔁 Reproducing the paper’s experiments

### Experiment 1 — Connectivity sweep (k=1…5)

```bash
# Example for all mesh types at low resolution (14/9) and high resolution (159/78)
bash experiments/exp1_connectivity/run_all.sh \
  --meshes U UC B F \
  --levels "14_9 159_78" \
  --k_list "1 2 3 4 5"
```

Outputs (per run):

* `metrics.json` with **global RMSE** per lead time (1–15 days).
* `rmse_map_15d.png`, `grad_map_15d.png` (|∇RMSE| seams).
* `kvoronoi_overlay.png` (empirical vs analytical order-k Voronoi boundaries).

### Experiment 2 — Node density sweep

```bash
bash experiments/exp2_density/run_all.sh \
  --meshes U UC B F \
  --sizes "14_9 20_9 34_20 52_27 159_78" \
  --k 3  # (or 4)
```

Outputs:

* RMSE vs lead time per size.
* RMSE vs nodes at fixed horizons (7, 10, 15 days).
* Error maps across sizes (structured vs bathymetry-aware).

---

## 📊 Expected results (sanity checks)

* **Best mean performance band:** **B-mesh** (k=3) and **F-mesh** (k=4) with **lower RMSE** and **fewer polygonal artifacts**.
* **Trend:** Increasing k beyond 4 **does not** consistently help; k=3–4 balances information flow and cost.
* **Structured meshes (U/UC):** adding nodes can **reduce input diversity** per mesh node and, with **uniform edge lengths**, produce **redundant edge embeddings** → **slightly worse long-horizon RMSE** and **amplified seams**.
* **Where to densify:** **coasts & steep-gradient regions**, where **finer-scale structures** must be captured.

*(Absolute numbers depend on hardware and exact data windows; relative trends should match the paper.)*

---

## ⚙️ Reproducibility tips

* Set seeds:

  ```bash
  export GNN_VORONOI_SEED=42
  ```
* Deterministic dataloaders and k-NN (KD-tree) are used by default.
* Logs: each run stores config, git commit hash, environment freeze file.

---

## 🛠️ Environment

```yaml
# environment.yml (excerpt)
name: gnn-voronoi-sst
channels: [conda-forge, pytorch]
dependencies:
  - python=3.10
  - pytorch
  - pytorch-cuda
  - numpy, scipy, pandas, xarray, netcdf4
  - scikit-learn, pyproj
  - matplotlib, cartopy
  - shapely
  - tqdm, pyyaml
  - pip
  - pip:
      - potpourri3d  # FPS utils (optional) or your FPS implementation
      - pykeops      # fast k-NN (optional)
```

---

## 📐 Figure regeneration

```bash
# Recreate all paper figures from stored results
python -m src.viz.make_all_figures \
  --runs_root runs/ \
  --out figures/
```

---

## 🧾 Citation

If you use this code or ideas, please cite:

```
Cuervo-Londoño, G.A.; Reyes, J.G.; Rodríguez-Santana, Á.; Sánchez, J.
Voronoi-Induced Artifacts from Grid-to-Mesh Coupling and Bathymetry-Aware
Meshes in Graph Neural Networks for Sea Surface Temperature Forecasting, 2025.
```

*(Add DOI/arXiv once available.)*

---

## 📝 License

MIT (see `LICENSE`).

---

## 🙌 Acknowledgments

ECOAQUA and CTIM (ULPGC) for computational support. Data from **Copernicus Marine**, **ECMWF ERA5**, and **NOAA ETOPO**.

---

## 📫 Contact

**Corresponding author:** Giovanny A. Cuervo-Londoño — `giovanny.cuervo@ulpgc.es`

---

## ✅ Repro checklist

* [ ] `conda activate gnn-voronoi-sst`
* [ ] `python -m src.data.download_all`
* [ ] `python -m src.data.preprocess`
* [ ] Build meshes (U, UC, B, F)
* [ ] Run **Exp1** (k-sweep) & **Exp2** (density sweep)
* [ ] `python -m src.viz.make_all_figures`

> Questions or issues? Open a GitHub issue.
