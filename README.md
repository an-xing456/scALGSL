# scALGSL: Active Learning and Graph Structure Learning for Cell Type Annotation from Single-Cell RNA-seq Data

**scALGSL** is a deep learning framework designed for the precise annotation of cell types and functional states in single-cell RNA sequencing (scRNA-seq) data. By integrating **Active Learning (AL)** and **Dynamic Graph Structure Learning (GSL)** within a **Graph Transformer** architecture, scALGSL effectively addresses batch effects, technical noise, and high-dimensional sparsity.

---

## Key Features
* **Active Learning Strategy**: Dynamically selects high-value "anchor" samples to enhance model generalization, particularly in small-sample and highly heterogeneous scenarios.
* **Dynamic Graph Structure Learning**: Iteratively optimizes cell-cell topological relationships and prunes noisy edges to ensure graph robustness.
* **Graph Transformer**: Captures complex, long-range cellular interactions to overcome data sparsity and heterogeneity.
* **Dual-path Feature Learning**: Simultaneously resolves cell identities and functional states, surpassing conventional single-classification approaches.

---

## Environment Requirement
The environment dependencies are managed via Conda. You can set up the environment using the provided `.yml` file.

```
# Create the environment from the yaml file
conda env create -f scALGSL_env.yml

# Activate the environment
conda activate scALGSL
```

## Data
### Wu2021 Dataset
Single-cell gene expression data with cell type annotations were sourced from the Single Cell Portal, specifically the Wu2021 dataset. This comprehensive dataset comprises 117,573 cells profiled using the Chromium 10X platform, spanning three cancer types (breast cancer, prostate cancer, and melanoma) across five subsets. All subsets contain Cancer cell populations. 

### CancerSEA (Functional States)
Cancer cell state annotations were obtained from CancerSEA, a curated database providing 14 functional states (including proliferation, stemness, and DNA repair activity scores) for 41,900 cells across 25 cancer types.

### Cross-platform PBMC Dataset
For cross-platform evaluation, we utilized the peripheral blood mononuclear cell (PBMC) dataset, containing six distinct sequencing platforms. Data were downloaded from the Broad Institute Single Cell Portal (https://portals.broadinstitute.org/single_cell/study/SCP424/single-cell-comparison-pbmc-data).

## Project Structure
```
scALGSL/
├── experiments/          # Stored raw and processed datasets (Wu2021, PBMC, etc.)
├── preprocess/           # Preprocessing scripts in Python and R
│   ├── h5ad2seurat_adaptor.py
│   └── preprocess.R
│   └── seurat2h5ad_adaptor.py
├── gt_main.py            # Main entry point for training and evaluation
├── scALGSL_env.yml       # Conda environment configuration
└── README.md             # Project documentation
```
