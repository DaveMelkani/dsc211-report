# Reconstructing Training Data from Trained Neural Networks
**DSC 211: Introduction to Optimization — Final Project**
*Dave Melkani & Eric Ness, UC San Diego*

This repository contains simulation code accompanying our review of [Haim et al. (2022)](https://arxiv.org/abs/2206.07758), which demonstrates that actual training samples can be reconstructed from the parameters of a trained neural network classifier alone, using the implicit bias of gradient flow and KKT conditions of the resulting maximum-margin problem.

---

## Repository Structure

```
dsc211-report/
├── Paper_Simulation.ipynb                                           # Main simulation notebook (Color Squares + MNIST)
├── Reconstructing Training Data from Trained Neural Networks.pdf    # Original paper
├── weights-mnist.pth                                                # Pre-trained MNIST binary MLP weights
├── color_square_example/                                            # Output images from Color Squares experiment
├── color_square_pth/                                                # Saved model weights from Color Squares experiment
├── mnist_pth/                                                       # Saved checkpoint tensors from MNIST experiment
└── data/                                                            # Raw MNIST data
```

---

## Simulations

The notebook `Paper_Simulation.ipynb` contains two proof-of-concept experiments that reproduce the core reconstruction pipeline described in the paper.

### 1. Color Squares Dataset (Synthetic Sanity Check)

A minimal synthetic binary classification dataset of 1×1 RGB pixel images — 10 "red" samples `(255, 0, 0)` and 10 "blue" samples `(0, 0, 255)` — is used to verify that the reconstruction pipeline is functioning correctly before scaling to real image data.

**Setup:**
- 3-layer fully-connected BinaryMLP (hidden widths: 1000), ReLU activations
- Modified backward pass: sigmoid substituted for the ReLU derivative to smooth the gradient landscape during reconstruction
- First layer includes a bias term; subsequent layers do not (preserving approximate homogeneity per Theorem 3.1)
- Trained to near-zero binary cross-entropy loss within 10 epochs

**Reconstruction:**
- 100 candidate inputs optimized (50 per class), initialized from 𝒩(0, 1)
- Dual variables initialized from Uniform(−5, 0)
- Two SGD optimizers with momentum: inputs (lr = 1e-5), dual variables (lr = 1e-4)
- 20,001 epochs; range loss penalizes values outside [0, 255]
- Total reconstruction loss drops from ~5,043 → 787 over training

A majority of reconstructed pixels approximate the red or blue target classes. Some intermediate colors (orange, magenta, purple) appear due to the nonconvexity of the reconstruction loss and sensitivity to initialization — consistent with the absence of uniqueness guarantees discussed in the paper.

---

### 2. MNIST Full Reconstruction

The full pipeline is applied to a binary MLP trained on a 500-sample subset of MNIST (250 odd-digit / 250 even-digit), following the experimental protocol of Haim et al.

**Setup:**
- Input images: 28×28, normalized by subtracting the per-pixel training mean
- Architecture: 784-1000-1000-1 MLP, same modified ReLU and homogeneous weight structure as above
- Pre-trained weights loaded from `weights-mnist.pth` (provided alongside the original paper); model set to eval mode before loading

**Reconstruction:**
- 1,000 candidate inputs, initialized with 𝒩(0, 1e-9) to avoid early ReLU saturation
- Dual variables initialized from Uniform(−10, 0)
- Range loss penalizes values outside [−1, 1] (normalized MNIST domain)
- 10,000 epochs; checkpoints saved every 1,000 epochs to `mnist_pth/`
- Total reconstruction loss: ~1.6×10⁵ → 376

**Important post-processing note:** The raw KKT optimization output is spatially incoherent and not directly interpretable. Adding back the per-pixel training mean (reversing normalization) is required before reconstructions become visually recognizable. This step is a prerequisite for any qualitative evaluation and is not emphasized in the original paper. After mean re-addition, reconstructed samples produce recognizable digit shapes (e.g., the numeral "3"), reproducing the core finding of Haim et al. at a smaller scale.

---
