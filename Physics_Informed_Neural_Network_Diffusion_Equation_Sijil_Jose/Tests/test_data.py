import torch
import os
import pytest

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flowde_pinn import gendata as data

def test_load_or_generate_gmm_1d(tmp_path):
    # temporary file path
    path = tmp_path / "gmm.pth"

    # generate new GMM
    X0, (K, PROB, MU, SIGMA) = data.load_or_generate_gmm_1d(
        path=path, N=1000, K=5, device="cpu"
    )

    # check tensor
    assert isinstance(X0, torch.Tensor)
    assert X0.shape == (1000, 1)
    assert X0.dtype == torch.float32

    # mixture params
    assert K == 5
    assert abs(sum(PROB) - 1.0) < 1e-6
    assert len(MU) == K
    assert len(SIGMA) == K

    # reload from saved file should give same K
    X0b, (Kb, _, _, _) = data.load_or_generate_gmm_1d(path=path, N=100, K=5)
    assert Kb == K


def test_fdist_matches_gaussians():
    # simple 2-component GMM
    PROB = [0.5, 0.5]
    import scipy.stats as st
    GAUSSIAN = [st.norm(0, 1), st.norm(2, 1)]

    xs = [-1.0, 0.0, 1.0, 2.0]
    vals = data.fdist(xs, PROB, GAUSSIAN)

    # should be non-negative and same length
    assert all(v >= 0 for v in vals)
    assert len(vals) == len(xs)


def test_plot1D_runs(tmp_path):
    # generate some data
    X = torch.randn(200, 1)

    # should run and create a figure
    filename = tmp_path / "plot.png"
    data.plot1D(X, f=None, filename=filename)

    assert filename.exists()

