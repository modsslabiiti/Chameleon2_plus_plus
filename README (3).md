# Chameleon2++

This repository contains the code and experiments from the manuscript:

**Chameleon2++: An Efficient Chameleon2 Clustering with Approximate Nearest Neighbors**  
📄 [Read the paper on arXiv](http://arxiv.org/abs/2501.02612)

---

## Overview

This project extends the original **Chameleon2** clustering algorithm with a more efficient version, **Chameleon2++**, using Approximate Nearest Neighbor (ANN) techniques. The included Jupyter notebooks enable:

- Reproducing results from the original **Chameleon2** paper.
- Running and analyzing **Chameleon2++**, the proposed improved method.
- Exploring different partitioning and ANN techniques.

---

## Repository Contents

The provided ZIP file contains **five Jupyter notebooks** divided into two main sections:

### 1. Chameleon2 (Original Implementation & Enhancements)

Reproduces and extends results from the original Chameleon2 paper:  
**Chameleon2: An Improved Graph-Based Clustering Algorithm**  
🔗 [ACM Link](https://dl.acm.org/doi/10.1145/3299876)

Partitioning methods used:
- **hMETIS**
- **FM-Bisection**

These notebooks provide detailed configuration options and improvements over the original implementation.

---

### 2. Chameleon2++ (Proposed Method)

Implements the **Chameleon2++** algorithm introduced in our manuscript.  
Key features:
- Uses **Approximate k-NN Graphs** for scalability.
- Leverages **hMETIS** for superior partitioning.

Supported ANN libraries:
- **Annoy** (performs best in our experiments)
- **FLANN**
- **NMSLIB**

---

## Installation & Dependencies

Install the following Python libraries before running the notebooks:

```bash
pip install numpy pandas networkx seaborn matplotlib tqdm
```

### Additional Tools

#### METIS (via `metis` Python wrapper)
- 🔗 [GitHub](https://github.com/KarypisLab/METIS)
- 📦 `pip install metis`
- 📚 [Python wrapper docs](https://metis.readthedocs.io/en/latest/)
- 🔧 [Issue #83 (for build help)](https://github.com/KarypisLab/METIS/issues/83)

#### Annoy (Spotify)
- 🔗 [GitHub](https://github.com/spotify/annoy)
- 📦 `pip install annoy`

#### FLANN
- 🔗 [GitHub](https://github.com/flann-lib/flann)
- 📦 `pip install flann`

#### NMSLIB
- 🔗 [GitHub](https://github.com/nmslib/nmslib)
- 📦 `pip install nmslib`  
  or  
  📦 `pip install --no-binary :all: nmslib` (if you encounter build issues)

---

## Reproducibility

To replicate the results:
- Refer to the [manuscript](http://arxiv.org/abs/2501.02612).
- Use the fine-tuned parameters provided in the manuscript.

---

## Citation

If you use this code in your research, please cite:

```
@article{singh2025chameleon2++,
  title={Chameleon2++: An Efficient Chameleon2 Clustering with Approximate Nearest Neighbors},
  author={Singh, Priyanshu and Ahuja, Kapil},
  journal={arXiv preprint arXiv:2501.02612},
  year={2025}
}
```