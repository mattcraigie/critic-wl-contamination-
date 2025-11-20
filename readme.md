# Critic-Based Conditional Dependence Detector for Weak Lensing Systematics

This repository provides an end-to-end implementation of a **contrastive critic method** to identify *unexpected causal relationships* in weak lensing shear measurements.

The goal is to detect **hidden contamination**: cases where some variable \(X\) (e.g., PSF ellipticity, seeing, chip position, cluster proximity) has a statistical relationship with the measured shear \(Y\), but where this dependence becomes visible **only within certain regions of the data**, represented by a learned low-dimensional variable \(T=f(Z)\).

This method generalizes to any system where you want to detect unexpected causal parents of a final output variable.

---

# Motivation

Weak lensing shear pipelines attempt to remove instrumental and observational systematics (PSF, seeing, blending, selection biases, focal-plane effects).  
However:

- Some systematics may not be globally visible,
- Their effect may appear **only in specific galaxy or instrument contexts**,
- Conventional global correlation checks easily miss these.

We want a **general, data-driven method** that automatically discovers:

1. Whether a variable \(X\) influences shear \(Y\),
2. In which context(s) that dependence becomes visible,
3. Without assuming any parametric form for the relationship.

This is exactly the kind of problem solved by a **critic-based conditional dependence estimator**.

---

# Core Idea

For each candidate systematic \(X\):

1. Treat shear as the output  
   \[
   Y = \hat\gamma
   \]

2. Treat all other features (galaxy, metadata, pipeline diagnostics) as contextual variables  
   \[
   Z
   \]

3. Learn a low-dimensional contextual coordinate  
   \[
   T = f_\phi(Z)
   \]

4. Train a **critic network**  
   \[
   D_\psi(X, Y, T)
   \]
   to distinguish between:
   - **Joint samples** \((X, Y, T)\)
   - **Independent samples** \((X, Y', T)\) where \(Y'\) is permuted

5. Using contrastive divergence bounds (DV / MINE / InfoNCE), estimate  
   \[
   I(X;Y|T)
   \]
   — the conditional mutual information.

6. Maximize this quantity with respect to the T-network:

\[
\max_\phi \max_\psi I(X;Y\mid T_\phi).
\]

If \(I(X;Y|T)\) is:

- **≈ 0** → no hidden dependence → X is clean  
- **≫ 0** → there exists a learned context \(T\) where X → Y dependence appears → potential contamination

This method is:

- Fully data-driven  
- Nonparametric  
- More efficient than learning full PDFs  
- Conceptually identical to classifier-based likelihood ratio estimation in simulation-based inference

---

# Toy Model: Hidden Contamination Example

We generate synthetic data where:

- The **global** joint distribution \(P(X,Y)\) shows **no visible correlation**,
- But conditioning on a contextual variable \(Z\) reveals **hidden dependence** introduced by a collider.

### Generative Process

For each sample:

X ~ Normal(0, 1)        # candidate systematic (e.g., PSF residual)
Y ~ Normal(0, 1)        # clean shear (independent of X)
Z = X + Y + Normal(0, 0.2)  # collider-like contextual variable


Interpretation:

- Marginally, `X` and `Y` are independent standard normals
- Conditioning on `Z` induces correlation because both `X` and `Y` influence `Z`
- Globally, P(X,Y) looks independent; given `Z`, dependence appears

This collider setup creates a clear contrast between global independence and conditional dependence.

---

# Repository Structure

critic-wl-contamination/
│
├── README.md # This file
├── data/
│ └── generate_toy_data.py # Script to generate synthetic dataset with hidden contamination
│
├── models/
│ ├── t_network.py # MLP: Z → T
│ ├── critic.py # Contrastive critic: (X, Y, T) → score
│ └── utils.py # Covariance helpers, etc.
│
├── training/
│ ├── train_global_dependence.py # Quick global X–Y dependence test (optional)
│ ├── train_conditional.py # Full T-network + critic joint training
│ └── losses.py # DV, InfoNCE, or NWJ objectives
│
├── experiments/
│ ├── run_toy_example.py # End-to-end toy run
│ └── visualize_results.ipynb # Plots of dependence vs T
│
└── requirements.txt # Pytorch + scientific stack


---

# Running the Toy Example

## 1. Install requirements

pip install -r requirements.txt


## 2. Generate the toy dataset

python data/generate_toy_data.py


This writes a `.npz` file with arrays for `X`, `Y`, and `Z`.

## 3. Train the critic + T-network

python experiments/run_toy_example.py


This script:

- Loads the toy dataset
- Trains the T-network and critic jointly
- Estimates conditional mutual information
- Prints final dependence scores

## 4. Visualize the results

Open the Jupyter notebook:

jupyter notebook experiments/visualize_results.ipynb


You'll see:

- **Global dependence** is near zero  
- **Conditional dependence given T** is strong  
- The learned T matches the true contextual Z up to a monotone transform  

---

# Interpreting Real Pipeline Results

On real weak lensing data:

- \(X\) is a candidate systematic (PSF size, ellipticity, chip-ID, seeing, cluster proximity, galaxy morphology, etc.)
- \(Y\) is the shear estimate (two components)
- \(Z\) includes all other information not being tested

A high conditional dependence score indicates:

> There exists a regime (learned by T) in which X influences shear Y.

This regime might correspond to:

- Poor PSF correction in certain focal-plane regions  
- Seeing dependence activating at high/low S/N  
- Blending effects for certain galaxy sizes  
- Selection effects that create collider bias  
- Redshift-dependent contamination  

This framework gives a **general, automated detector** for unexpected causal parents of shear.

---

# Summary

This repository implements a modern, flexible method for discovering **hidden contamination pathways** in weak lensing shear measurements:

- Uses a critic to approximate likelihood ratios  
- Uses a learned contextual coordinate T to reveal complex conditional dependence  
- Detects subtle systematics invisible in global statistics  
- Scales to high-dimensional context  
- Easily extendable to real weak lensing pipelines  
