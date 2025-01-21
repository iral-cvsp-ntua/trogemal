# TROGEMAL: Tropical Geometry and Machine Learning

## Overview

**TROGEMAL** (Tropical Geometry and Machine Learning) is a research project led by **Professor Petros Maragos** at the **National Technical University of Athens**. The project focuses on leveraging tropical geometry and max-plus algebra to advance the theoretical analysis and development of machine learning systems and algorithms.

This three-year initiative explores innovative methodologies in machine learning, including applications in neural networks, graphical models, and nonlinear regression, aiming to contribute both to theoretical advancements and practical applications in artificial intelligence.

[Visit the Project Website](https://robotics.ntua.gr/trogemal/)

---

## Objectives

The project focuses on four primary research directions:

1. **Developing New Tropical Regression Techniques**  
   - Creating methods for data fitting using tropical polynomials and piecewise-linear (PWL) functions.  
   - Applications include multivariate nonlinear regression.

2. **Analyzing and Simplifying Neural Networks**  
   - Leveraging tropical geometry to understand and reduce the complexity of neural networks, focusing on piecewise-linear activation functions.

3. **Enhancing Graphical Models and Inference Algorithms**  
   - Improving algorithms such as the Viterbi algorithm and probabilistic graphical models through tropical geometry techniques.

4. **Extending Tropical Geometry**  
   - Developing a generalized max-* algebra to address machine learning problems over weighted lattices.

---

## Project Structure

```plaintext
.
├── sample_polytopes.py
├── tropic_hannah_surface_50MMAE.pdf
├── tropic_plane_MMAE.pdf
├── tropical_fit_hoburg_toy_6lines.pdf
└── tropical_project
    ├── hannah
    │   ├── data.txt
    │   ├── tropical_regression_hannah.py
    │   └── tropical_regression_hannah_gradient.py
    ├── hoburg
    │   ├── data.txt
    │   ├── tropical_regression_hoburg.py
    │   └── tropical_regression_hoburg_jenks.py
    └── tropical_regression.py
```

---

## Key Components

### 1. **`sample_polytopes.py`**
A utility script for sampling vertices from the convex hull of a Minkowski sum of polytopes. Implements Algorithm 1 from [this paper](https://arxiv.org/abs/1805.08749).  
**Features**:
- Ensures compatibility of polytope dimensions.
- Samples convex hull vertices using Gaussian sampling.

**Example Usage**:
```python
from sample_polytopes import sample_convex_hull_vertices

# Define polytopes
polytope1 = np.random.rand(10, 3)
polytope2 = np.random.rand(15, 3)

# Sample vertices
vertices = sample_convex_hull_vertices(100, [polytope1, polytope2])
print(vertices)
```

---

### 2. **`tropical_regression.py`**
A general-purpose tropical regression script.  
- **Output**: Generates one-sided regression plots, e.g., `tropic_plane_MMAE.pdf`.

---

### 3. **`tropical_regression_hoburg.py`**
Implements tropical regression based on [this paper](https://link.springer.com/article/10.1007/s11081-016-9332-3).  
- Two variants:
  - Vanilla regression.
  - K-Means (Jenks breaks) for slope detection.
- **Output**: Plots such as `tropical_fit_hoburg_toy_6lines.pdf`.

---

### 4. **`tropical_regression_hannah.py`**
Implements tropical regression based on [this paper](https://arxiv.org/abs/1105.1924).  
- Two variants:
  - Vanilla regression.
  - K-Means for slope detection.
- **Output**: Surface plots, e.g., `tropic_hannah_surface_50MMAE.pdf`.

---

## Applications

### Tropical Geometry for Neural Networks
The project's tropical geometry framework can be extended to analyze **linear regions** of neural networks. By modifying the `sample_polytopes.py` script to sample activation regions, users can:

1. Represent neural network activation regions as polytopes.
2. Count and analyze linear regions to understand network complexity and decision boundaries.

**Proposed Steps**:
1. Extract activation outputs for a set of inputs.
2. Represent activations as polytopes using their corresponding weights and biases.
3. Use `sample_convex_hull_vertices` to enumerate and analyze the linear regions.

---


## License

This project is licensed under the [MIT License](LICENSE).

---

For more information, please visit the [TROGEMAL website](https://robotics.ntua.gr/trogemal/).
