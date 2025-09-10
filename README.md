# Approximate-Chameleon: Approx-Ch

This repository contains the code and experiments from the manuscript:

**Approx-Ch: An Approximate-Chameleon Clustering for Large-Scale and High-Dimensional Data**  
📄 [Read the paper on arXiv](http://arxiv.org/) # Insert/Update arxiv link here (once uploaded)

---

## Overview

This project propose an **Approximate-Chameleon** clustering algorithm for Large-Scale and High-Dimensional Data, **Approx-Ch**, using Approximate Nearest Neighbor (ANN) techniques. The included python code enable:

- Reproducing results from the original **Chameleon2** paper - using Ch2.py.
- Running and analyzing **Approximate-Chameleon**, the proposed improved method - using Approx-Ch.py.

---

## Repository Contents

The provided ZIP file contains **two set of python codes** addressing Ch2 and Approx-Ch:

### 1. Chameleon2 (Original Implementation & Enhancements): Ch2.py

Reproduces and extends results from the original Chameleon2 paper:  
**Chameleon2: An Improved Graph-Based Clustering Algorithm**  
🔗 [ACM Link](https://dl.acm.org/doi/10.1145/3299876)

- Uses traditional **exact k-NN Graph**.
Partitioning methods used:
- **Recursive FM-Bisection**

The code provide detailed configuration options over the original implementation.

---

### 2. Approximate-Chameleon (Proposed Method): Approx-Ch.py

Implements the **Approximate-Chameleon** algorithm introduced in our manuscript.  
Key features:
- Uses **Approximate k-NN Graphs** for scalability.
- Leverages **hMETIS** for superior partitioning.

Supported ANN libraries:
- **Annoy**

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

---

## Reproducibility

To replicate the results:
- Refer to the [manuscript](http://arxiv.org/). # Insert arxiv link here (once uploaded)
- Use the fine-tuned parameters provided in the manuscript.

---

## Citation

If you use this code in your research, please cite: # Insert/Update the citation once arxiv is available

```
@article{
}
```