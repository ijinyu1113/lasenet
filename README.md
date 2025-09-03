# Latent Variable Sequence Modeling

**Research conducted at the CognAc Lab, UC Berkeley Department of Cognitive Neuroscience.**

---

## Project Description

This project focuses on a core challenge in cognitive modeling: learning the latent variables of **volatility** and **stochasticity** from a sequence of perceptual data. My primary contribution is the development of a comprehensive computational framework for this task. I implemented **Kalman filtering** and **particle filtering** techniques in MATLAB to simulate latent variables, creating the necessary ground truth data for model training and validation. The resulting simulation data is then used to train and evaluate neural network models.

For this purpose, I converted the original TensorFlow-based LaseNet model into a PyTorch implementation. I also developed and implemented a novel **Transformers-based architecture** to compare its performance against the PyTorch RNN.

---

## Data & Methodological References

The volatility and stochasticity simulation data used in this work is based on the model described in the following paper:

* Piray, P., & Daw, N. D. (2021). **A model for learning based on the joint estimation of stochasticity and volatility**. *Nature Communications*, 12(1), 6587.

The original paper can be accessed [here](https://drive.google.com/file/d/1B7J4jFpU-3d2p81y8yL5y4l_fV6Xw9f0/view?usp=sharing).

The RNN methodology is inspired by the **LaseNet** framework, described in:

* Pan, T., Li, J., Thompson, B., & Collins, A. (2025). **LaseNet: Latent Variable Sequence Identification for Cognitive Models with Neural Network Estimators**. *Behav Res*, 57, 272.

---

## Repository Structure

* `notebooks/`: Contains the Jupyter notebooks for training, analysis, and model comparison.
* `src/`: Holds the Python source code, including the model implementations.
* `matlab/`: Contains the MATLAB code for latent variable simulation, including `sim_uniform_LVLS.m`.

---

## Results

To evaluate my models, I simulated data, fitted the models to it, and then reconstructed a person's perception from the inferred volatility and stochasticity. Preliminary results from the `model_comparison.ipynb` notebook show that both the RNN and Transformer models perform very well, with the Transformer model achieving slightly better performance.

However, the current model has a problem with fitting human bias in perception. The research on this issue and the validation of the models' performance on real human data is ongoing.

## Acknowledgements

I would like to thank my PhD mentor, Tianhe Wang, and Ti-Fen Pan for their guidance on this project.