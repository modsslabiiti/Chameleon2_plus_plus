# Chameleon2++

This repository contains the code and experiments from the manuscript:

**Chameleon2++: An Efficient and Scalable Variant Of Chameleon Clustering**  
📄 [Read the paper on arXiv](http://arxiv.org/abs/2501.02612)

---

## Overview

This project extends the original **Chameleon2** clustering algorithm with an efficient and scalable version, **Chameleon2++**, using Approximate Nearest Neighbor (ANN) techniques. The included python code enable:

- Reproducing results from the original **Chameleon2** paper - using Ch2.py.
- Running and analyzing **Chameleon2++**, the proposed improved method - using Ch2++.py.

---

## Repository Contents

The provided ZIP file contains **two set of python codes** addressing Ch2 and Ch2++:

### 1. Chameleon2 (Original Implementation & Enhancements): Ch2.py

Reproduces and extends results from the original Chameleon2 paper:  
**Chameleon2: An Improved Graph-Based Clustering Algorithm**  
🔗 [ACM Link](https://dl.acm.org/doi/10.1145/3299876)

- Uses traditional **exact k-NN Graph**.
Partitioning methods used:
- **Recursive FM-Bisection**

The code provide detailed configuration options over the original implementation.

---

### 2. Chameleon2++ (Proposed Method): Ch2++.py

Implements the **Chameleon2++** algorithm introduced in our manuscript.  
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
- Refer to the [manuscript](http://arxiv.org/abs/2501.02612).
- Use the fine-tuned parameters provided in the manuscript.

---

## Citation

If you use this code in your research, please cite: # Insert/Update the citation once arxiv is available

```
@article{singh2025chameleon2++,
  title={Chameleon2++: An efficient chameleon2 clustering with approximate nearest neighbors},
  author={Singh, Priyanshu and Ahuja, Kapil},
  journal={arXiv preprint arXiv:2501.02612},
  year={2025}
}
```
