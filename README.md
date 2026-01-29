# DupliMend: Online Detection and Refinement of Imprecise Activity Labels

[![Paper](https://img.shields.io/badge/Paper-CAiSE'26-blue)]()
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Source code and additional resources for the paper **"DupliMend: Online Detection and Refinement of Imprecise Activity Labels"** by Savandi Kalukapuge, Andrzej Janusz, and Moe Thandar Wynn (CAiSE 2026).

## Key Contributions

1. **A streaming, unsupervised framework that detects and refines homonymous activity labels on-the-fly without prior specification, complete traces, or full-log analysis.**

2. **Activity-specific sparse denoising autoencoders that learn multi-perspective representations and enable dynamic splitting and merging of label variants as behaviour evolves through online machine learning, specifically, online clustering.**

3. **A drift-aware continual learning mechanism combining ADWIN, cluster regularisation, and centroid memory replay to maintain stability and prevent forgetting under evolving process event streams.**

## Approach Overview
![DupliMend High-Level Overview](DupliMend_Approach_Overview.png)

## Quick Start

Run DupliMend on the included I-PALIA dataset and reproduce evaluation metrics.

### Step 1: Clone and Install

```bash
git clone https://github.com/Savandi/DupliMend.git
cd DupliMend
python -m venv venv
source venv/bin/activate  # Linux/Mac (or: venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### Step 2: Run DupliMend

The repository includes the **I-PALIA** synthetic dataset:
- Input: `src/synthetic_logs/ipalia.csv` (~8,400 events)
- Ground truth: `src/synthetic_logs/ipalia_groundtruth.csv`

This dataset contains a homonymous activity label **"A"** that should be refined into three variants: `A_start`, `A_middle`, and `A_end`.

```bash
python main.py
```

This will process events in streaming fashion, train activity-specific autoencoders online, cluster events, and output results to `run_output/tracking_YYYYMMDD_HHMMSS/`.

### Step 3: Evaluate Results

After `main.py` completes, evaluate clustering quality against ground truth:

```bash
# Set the tracking directory (replace with actual output folder name)
export TRACKING_DIR=run_output/tracking_YYYYMMDD_HHMMSS  # Linux/Mac
# $env:TRACKING_DIR="run_output\tracking_YYYYMMDD_HHMMSS"  # Windows PowerShell

cd src/evaluation_scripts
python evaluate_single_test_file.py
```

Results are saved to `evaluation_results/`.

---

## Evaluation Metrics

**Clustering Quality (Supervised):**
- **ARI** (Adjusted Rand Index) - alignment with ground truth, adjusted for chance
- **NMI** (Normalised Mutual Information) - information-theoretic cluster-label agreement
- **Expected Entropy (Cluster)** - entropy of ground truth labels within clusters
- **Expected Entropy (Label)** - entropy of cluster assignments per ground truth label

**Clustering Quality (Unsupervised):**
- **Silhouette Score** - cohesion and separation measure

**Discovered Model Quality:**
- **Log Fitness** (Recall) - fraction of log behaviour reproducible by the model
- **Log Precision** - fraction of model behaviour observed in the log
- **F-score** - harmonic mean of Precision and Fitness

**Streaming Performance:**
- Convergence speed at intermediate checkpoints (25%, 50%, 75%, 100%)
- Throughput (events/second), Latency (ms/event), Peak Memory (MB)

---

## Key Experimental Results

**Clustering quality (ARI) on synthetic PESs**

![ARI comparison across methods and datasets](experimental_results/ari_boxplot_comparison.png)

**Discovered model precision across all PESs**

![Precision for DupliMend vs baselines](experimental_results/precision_all_categories.png)

**Statistical comparison of precision (Friedman + Nemenyi)**
- Using the Friedman test with Nemenyi post-hoc analysis (α = 0.05), **DupliMend achieves the best mean rank (1.43)** for Precision and is the only method that differs significantly from the unrefined baseline.
![Critical difference diagram for precision](experimental_results/cd_diagram_logprecision_9.png)

**Precision convergence over the stream (real-life PESs)**

![Precision convergence on real-life datasets](experimental_results/precision_convergence_reallife_datasets.png)

Full experimental results: [Dropbox Link](https://www.dropbox.com/scl/fo/qfxvyagrczl68hiq1xd17/ANasY83Ti-5otlTkbnrYmW8?rlkey=yocrf8hy63euj8hwodfq2kids&st=e2ng9b3v&dl=0)

Optimised hyperparameters from Bayesian optimisation: `src/bayesian_optimization/`

---

## Datasets

### Synthetic Process Event Streams

| Dataset | Events | Cases | Activities | Description |
|---------|--------|-------|------------|-------------|
| **DuplicatedTasks** | ~1,000/log | 1,000/log | varies | 1,295 synthetic logs from [4TU](https://data.4tu.nl/articles/_/12718226/1) with injected homonyms |
| **I-PALIA** | 8,415 | 990 | 8 (10 GT) | Healthcare workflow with duplicated activity "A" (accessible in the [synthetic logs folder](./src/synthetic_logs/)) | 
| **DocReview** | 158,855 | 20,090 | 10 (11 GT) | Document review workflow with multi-attribute events (accessible in the [synthetic logs folder](./src/synthetic_logs/)) | 

### Real-Life Process Event Streams

These event logs are obtained from [4TU.ResearchData](https://data.4tu.nl/), the standard repository for publicly available process mining datasets.

| PES | Events | Cases | Activities | Avg Case Length | Publicly Available Link |
|-----|--------|-------|------------|-----------------|-----|
| **BPIC2012** | 262,200 | 13,087 | 36 | 20 | [BPIC Challenge 2012](https://data.4tu.nl/articles/_/12689204/1) |
| **BPIC2013C** | 6,660 | 1,487 | 7 | 4 | [BPI Challenge 2013, closed problems](https://data.4tu.nl/datasets/1987a2a6-9f5b-4b14-8d26-ab7056b17929/1) |
| **BPIC2013I** | 65,533 | 7,554 | 13 | 9 | [BPI Challenge 2013, incidents](https://data.4tu.nl/articles/_/12693914/1) |
| **BPIC2013O** | 2,351 | 819 | 5 | 3 | [BPI Challenge 2013, open problems](https://data.4tu.nl/articles/_/12688556/1) |
| **BPIC2017** | 1,202,267 | 31,509 | 26 | 38 | [BPI Challenge 2017](https://data.4tu.nl/articles/_/12696884/1) |
| **Road Fines** | 561,470 | 150,370 | 11 | 4 | [Road Traffic Fine Management Process](https://data.4tu.nl/articles/_/12683249/1) |
| **Env. Permits** | 8,577 | 1,434 | 27 | 6 | [Receipt phase of an environmental permit application process (WABO), CoSeLoG project](https://data.4tu.nl/articles/_/12709127/2) |

---

## Baseline Methods

DupliMend is compared against two established offline methods:

| Method | Reference | Algorithm |
|--------|-----------|-----------|
| **lblrefine** | Lu et al. (BPM 2016) | Connected-components on event graph |
| **lblsplit** | van Zelst et al. (BPM 2023) | Leiden community detection with variant compression |

---

## Results Summary

### Streaming Performance

| PES | Latency (ms/ev) | Peak Memory (MB) | Throughput (ev/s) |
|-----|-----------------|------------------|-------------------|
| DuplicatedTasks | 19.10 | 50.24 | ~52 |
| I-PALIA | 16.42 | 36.72 | ~61 |
| BPIC2012 | 49.97 | 98.52 | ~20 |
| BPIC2017 | 42.80 | 82.36 | ~23 |
| Road Fines | 33.50 | 65.18 | ~30 |

### Clustering Quality (Synthetic PESs)

| Dataset | ARI | NMI | EE Cluster | EE Label |
|---------|-----|-----|------------|----------|
| DuplicatedTasks (avg) | 0.96 | 0.94 | 0.08 | 0.06 |
| I-PALIA | 0.92 | 0.89 | 0.12 | 0.10 |
| DocReview | 0.94 | 0.91 | 0.09 | 0.08 |

### Process Model Quality Improvement

| PES | Method | Fitness | Precision | F-score |
|-----|--------|---------|-----------|---------|
| Road Fines | Unrefined | 0.85 | 0.62 | 0.72 |
| Road Fines | **DupliMend** | 0.88 | **0.78** | **0.83** |
| BPIC2017 | Unrefined | 0.79 | 0.58 | 0.67 |
| BPIC2017 | **DupliMend** | 0.82 | **0.71** | **0.76** |

---

## Acknowledgments

This work was conducted at the Queensland University of Technology (QUT), School of Information Systems in the Faculty of Science and Centre for Data Science.

## References

1. Lu, X., Fahland, D., van den Biggelaar, F. J. H. M., & van der Aalst, W. M. P. (2016). *Handling duplicated tasks in process discovery by refining event labels.* In BPM 2016 (LNCS 9850, pp. 90–107). Springer. [DOI](https://doi.org/10.1007/978-3-319-45348-4_6)

2. van Zelst, S. J., Tai, J., Langenberg, M., & Lu, X. (2023). *Context-based activity label-splitting.* In BPM 2023 (LNCS 14159, pp. 232–248). Springer. [DOI](https://doi.org/10.1007/978-3-031-41620-0_14)
