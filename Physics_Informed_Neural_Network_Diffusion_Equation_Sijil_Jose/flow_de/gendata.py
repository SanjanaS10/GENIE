import os, sys

# standard module for array manipulation
import numpy as np

# standard module for high-quality plots
import matplotlib.pyplot as plt
# standard module for machine learning
import torch
import torch.nn as nn

from tqdm import tqdm

import scipy.stats as st
import joblib
import matplotlib.patches as mpatches



def load_or_generate_gmm_1d(path="X01d.pth", N=50000, K=20, device="cpu"):
    """
    Load a 1D Gaussian Mixture Model from file, or generate a new one.
    Returns:
        X0: torch.Tensor of shape (N,1)
        (K, PROB, MU, SIGMA): mixture parameters
    """
    try:
        K, PROB, MU, SIGMA = joblib.load(path)
    except FileNotFoundError:
        PROB = np.random.uniform(0, 1, K)
        PROB /= PROB.sum()
        MU = np.random.uniform(-4.5, 4.5, K)
        SIGMA = np.random.uniform(0.05, 0.3, K)
        joblib.dump([K, PROB, MU, SIGMA], path)

    # pick components
    k = np.random.choice(np.arange(K), p=PROB, size=N)
    mu = MU[k]
    sigma = SIGMA[k]
    X0 = np.random.normal(mu, sigma)

    X0 = torch.tensor(X0, dtype=torch.float32, device=device).view(-1, 1)
    return X0, (K, PROB, MU, SIGMA)

def fdist(x, PROB, GAUSSIAN):
    """Evaluate PDF of GMM at points x."""
    return np.array([p * f.pdf(x) for p, f in zip(PROB, GAUSSIAN)]).sum(axis=0)

def plot1D(x, f=None, xbins=200, xmin=-5, xmax=5, ymax=0.6, filename=None):
    """Histogram + analytic PDF overlay."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title('Target Density $p(x_0)$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")

    try:
        x = x.cpu()
    except:
        pass

    ax.hist(x, bins=xbins, range=[xmin, xmax],
            color="steelblue", density=True, alpha=0.4)

    if f:
        xx = np.linspace(xmin, xmax, 2*xbins+1)
        yy = f(xx)
        ax.plot(xx, yy, color="red")

    fig.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()



def load_or_generate_gmm_2d(path="X02d.pth", N=50000, K=20, device="cpu"):
    """
    Load a 2D Gaussian Mixture Model from file, or generate a new one.
    Returns:
        X0: torch.Tensor of shape (N,2)
        (K, PROB, MU, SIGMA): mixture parameters
    """
    try:
        K, PROB, MU, SIGMA = joblib.load(path)
    except FileNotFoundError:
        # Generate new 2D GMM
        PROB = np.random.uniform(0, 1, K)
        PROB /= PROB.sum()  # Normalize to sum to 1

        # Means: Randomly sampled in [-4.5, 4.5] for both dimensions
        MU = np.random.uniform(-4.5, 4.5, (K, 2))

        # Covariance matrices: Random but symmetric positive-definite
        SIGMA = []
        for _ in range(K):
            A = np.random.uniform(-0.5, 0.5, (2, 2))
            SIGMA_i = A @ A.T + np.eye(2) * 0.1  # Ensure positive-definite
            SIGMA.append(SIGMA_i)
    
        # Save GMM parameters
        joblib.dump([K, PROB, MU, SIGMA], path)

    # pick components
    k = np.random.choice(K, size=N, p=PROB)  # Choose components
    samples = np.array([np.random.multivariate_normal(MU[i], SIGMA[i]) for i in k])
    
    
    X0 = torch.tensor(samples, dtype=torch.float32).to(device)
    return X0, (K, PROB, MU, SIGMA)

def load_or_generate_gmm_3d(path="X03d.pth", N=50000, K=20, device="cpu"):
    """
    Load a 2D Gaussian Mixture Model from file, or generate a new one.
    Returns:
        X0: torch.Tensor of shape (N,2)
        (K, PROB, MU, SIGMA): mixture parameters
    """
    try:
        K, PROB, MU, SIGMA = joblib.load(path)
    except FileNotFoundError:
        # Generate new 2D GMM
        PROB = np.random.uniform(0, 1, K)
        PROB /= PROB.sum()  # Normalize to sum to 1

        # Means: Randomly sampled in [-4.5, 4.5] for both dimensions
        MU = np.random.uniform(-4.5, 4.5, (K, 3))

        # Covariance matrices: Random but symmetric positive-definite
        SIGMA = []
        for _ in range(K):
            A = np.random.uniform(-0.5, 0.5, (3, 3))
            SIGMA_i = A @ A.T + np.eye(3) * 0.1  # Ensure positive-definite
            SIGMA.append(SIGMA_i)

        # Save GMM parameters
        joblib.dump([K, PROB, MU, SIGMA], path)

    # pick components
    k = np.random.choice(K, size=N, p=PROB)  # Choose components
    samples = np.array([np.random.multivariate_normal(MU[i], SIGMA[i]) for i in k])


    X0 = torch.tensor(samples, dtype=torch.float32).to(device)
    return X0, (K, PROB, MU, SIGMA)


