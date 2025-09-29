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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flowde_pinn import gendata
from flowde_pinn import flow_de
from flowde_pinn import networks_1d

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Training on GPU.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Training on CPU.")


class PhysicsNN(nn.Module):
    def __init__(self, x0, z, sigma0=1e-2):
        super(PhysicsNN, self).__init__()
        self.q = flow_de.qVectorField(x0, debug=False).to(device)
        self.z = z.to(device)
        
        # More stable architecture
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        
        # Initialize weights properly
        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.normal_(m.bias, mean=0, std=0.1)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        x = torch.cat([t, z], dim=1)
        
        return self.net(x)

def set_seed(seed=42):
    '''
    Seeding the random variables for reproducibility
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def x_function(model: PhysicsNN, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    t_1 = torch.ones_like(t)

    out1 = model(t, z)
    out2 = model(t_1, z)

    if not torch.isfinite(out1).all():
        print("model(t, z) has NaN or Inf")
        print("t =", t)
        print("z =", z)
        print("model(t, z) =", out1)
        raise ValueError("model(t, z) returned NaN or Inf")

    if not torch.isfinite(out2).all():
        print("model(1, z) has NaN or Inf")
        print("t_1 =", t_1)
        print("z =", z)
        print("model(t_1, z) =", out2)
        raise ValueError("model(t_1, z) returned NaN or Inf")

    out = z + out1 - out2

    if not torch.isfinite(out).all():
        print("Final x_function output has NaN or Inf")
        print("z =", z)
        print("model(t, z) =", out1)
        print("model(1, z) =", out2)
        print("x =", out)
        raise ValueError("x_function returned NaN or Inf")

    return out


def derivative(model: PhysicsNN, t_f: torch.Tensor, z: torch.Tensor ,order: int = 1) -> torch.Tensor:
    """
    This function calculates the derivative of the Anzat at t_f
    """
    t_f.requires_grad_(True)
    dy = model(t_f, z)
    for i in range(order):
        dy = torch.autograd.grad(
            dy, t_f, grad_outputs = torch.ones_like(t_f), create_graph=True, retain_graph=True
        )[0]
    return dy


def f(model: PhysicsNN, t_f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    This function evaluates the ODE governing the reverse flow at times t for inputs z
    
    """
    x = x_function(model, t_f,z)
    dxdt = derivative(model, t_f, z, order = 1)
    q_val = model.q.G(t_f, x)
    if not (torch.isfinite(dxdt).all() and torch.isfinite(q_val).all()):
        raise ValueError("NaN in derivative or q.G")    
    f = t_f*dxdt - t_f*q_val
    return f

def loss_function(model: PhysicsNN, t_f: torch.Tensor, z : torch.Tensor, t_u: torch.Tensor, z_u:torch.Tensor, x_u:torch.Tensor) -> torch.Tensor:
    # Loss associated with the physics governing the model
    MSE_f = f(model, t_f, z).pow(2).sum()

    #Loss associated with training data
    MSE_u = (x_function(model, t_u, z_u)- x_u).pow(2).sum()
    return MSE_u + MSE_f*10

def loss_function_physics(model: PhysicsNN, t_f: torch.Tensor, z : torch.Tensor) -> torch.Tensor:
    # Loss associated with the physics governing the model
    MSE_f = f(model, t_f, z).pow(2).mean()

    return MSE_f

iter = 0  # Make sure `iter` is defined globally at module level

def closure(model: PhysicsNN, t_f: torch.Tensor, z: torch.Tensor,
            t_u: torch.Tensor, z_u: torch.Tensor, x_u: torch.Tensor) -> torch.Tensor:
    """
    Closure function required by LBFGS optimizer.
    """
    #model.load_state_dict(torch.load('/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters.pth'))
    global iter
    model.zero_grad()  #  Move this line *inside* the closure, not passed in
    loss = loss_function(model, t_f, z, t_u, z_u, x_u)

    if torch.isnan(loss) or torch.isinf(loss):
        print("Closure encountered NaN or Inf in loss.")
        return torch.tensor(0.0, requires_grad=True).to(DEVICE)

    loss.backward()

    iter += 1
    if iter % 1000 == 0:
        print(f"Iteration: {iter}, Loss: {loss.item()}")
        torch.save(model.state_dict(), '/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters_2.pth')
    #print(f"Iteration: {iter}, Loss: {loss.item()}")
    return loss

from functools import partial
from torch.utils.data import TensorDataset, DataLoader

def train(model: PhysicsNN, z_full, t_u, z_u, x_u, ramp_steps=5, initial_z_frac=0.1,lr_target=1.0,lr_warmup_factor=0.1):
    """
    Trains the model using the LBFGS optimizer with closure.
    """
    #model.load_state_dict(torch.load('/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters_2.pth'))

    # Training data grid
    T_u, Z_u = np.meshgrid(t_u, z_u)
    T_u_flat = torch.Tensor(T_u).flatten()[:, None].float().to(DEVICE)
    Z_u_flat = torch.Tensor(Z_u).flatten()[:, None].float().to(DEVICE)
    X_u_flat = torch.tensor(x_u.T[0]).flatten()[:, None].float().to(DEVICE)
    del T_u, Z_u

    t = np.linspace(0, 1, 100)
    total_z = len(z_full)
    z_sizes = [int(initial_z_frac * total_z + (i / (ramp_steps - 1)) * (total_z - initial_z_frac * total_z)) for i in range(ramp_steps)]
    lrs = [ (lr_target + 1e-5 )- lr_target * (lr_warmup_factor + (i / (ramp_steps - 1)) * (1 - lr_warmup_factor)) for i in range(ramp_steps)]
    for i, (z_size, lr) in enumerate(zip(z_sizes, lrs)):
        z_subset = z_full[:z_size]
        T_colloc, Z_colloc = np.meshgrid(t, z_subset)


        T_flat = torch.Tensor(T_colloc).flatten()[:, None].float()
        Z_flat = torch.Tensor(Z_colloc).flatten()[:, None].float()
        colloc_dataset = TensorDataset(T_flat, Z_flat)
        colloc_loader = DataLoader(colloc_dataset, batch_size=131072, shuffle=True)
        del T_colloc, Z_colloc

        print(f"\n Ramp Step {i+1}/{ramp_steps} — z_size: {z_size}, learning rate: {lr:.5f}")

        # Create LBFGS optimizer
        optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr= lr,
                                  max_iter=10000,
                                  max_eval=10000,
                                  history_size=200,
                                  tolerance_grad=1e-100,
                                  tolerance_change=1e-100,
                                  line_search_fn="strong_wolfe")


        for batch_idx, (T_batch, Z_batch) in enumerate(colloc_loader):
            T_batch, Z_batch = T_batch.to(DEVICE), Z_batch.to(DEVICE)
            closure_fn = lambda: closure(model, T_batch, Z_batch, T_u_flat, Z_u_flat, X_u_flat)
            optimizer.step(closure_fn)



        #  Fix: Don't pass optimizer to closure — it's not required by PyTorch
        #closure_fn = lambda: closure(model, T_flat, Z_flat, T_u_flat, Z_u_flat, X_u_flat)
        torch.save(model.state_dict(), '/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters_2.pth')
        # Run LBFGS optimization loop
        optimizer.step(closure_fn)

    return model, []  # loss_list was unused in this LBFGS form

from functools import partial
def train_adam(model: PhysicsNN, z_full, t_u, z_u, x_u, ramp_steps=5, initial_z_frac=0.1,lr_target=1.0,lr_warmup_factor=0.1):
    """
    Trains the model using the Adam Optimizer.
    """
    #model.load_state_dict(torch.load('/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters_2.pth'))
    epoch = 5000
    # Training data grid
    T_u, Z_u = np.meshgrid(t_u, z_u)
    T_u_flat = torch.Tensor(T_u).flatten()[:, None].float().to(DEVICE)
    Z_u_flat = torch.Tensor(Z_u).flatten()[:, None].float().to(DEVICE)
    X_u_flat = torch.tensor(x_u.T[0]).flatten()[:, None].float().to(DEVICE)
    del T_u, Z_u

    loss_list = []
    t = (np.linspace(0, 1, 100))
    total_z = len(z_full)
    z_sizes = [int(initial_z_frac * total_z + (i / (ramp_steps - 1)) * (total_z - initial_z_frac * total_z)) for i in range(ramp_steps)]
    lrs = [ (lr_target + 1e-5 )- lr_target * (lr_warmup_factor + (i / (ramp_steps - 1)) * (1 - lr_warmup_factor)) for i in range(ramp_steps)]
    for i, (z_size, lr) in enumerate(zip(z_sizes, lrs)):
        z_subset = z_full[:z_size]
        T_colloc, Z_colloc = np.meshgrid(t, z_subset)
        T_flat = torch.Tensor(T_colloc).flatten()[:, None].float()
        Z_flat = torch.Tensor(Z_colloc).flatten()[:, None].float()
        #colloc_dataset = TensorDataset(T_flat, Z_flat)
        #colloc_loader = DataLoader(colloc_dataset, batch_size=131072, shuffle=True)
        del T_colloc, Z_colloc

        colloc_dataset = TensorDataset(T_flat, Z_flat)
        colloc_loader = DataLoader(colloc_dataset, batch_size=131072, shuffle=True)

        print(f"\n Ramp Step {i+1}/{ramp_steps} — z_size: {z_size}, learning rate: {lr:.5f}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epoch):
            for batch_idx, (T_batch, Z_batch) in enumerate(colloc_loader):
                T_batch, Z_batch = T_batch.to(DEVICE), Z_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_function(model, T_batch,  Z_batch, T_u_flat, Z_u_flat, X_u_flat )
            loss.backward()
            optimizer.step()
            if (epoch % 1000 == 0) and epoch != 0:
              loss_list.append(loss.item())
              # Sample collocation points
              #t_eval = torch.rand(1000, 1)
              #z_eval = get_normal_sample(t_eval).to(DEVICE)
                # Evaluation part
                # Evaluate the model
              #eval_loss = loss_function_physics(model,t_eval.to(device) , z_eval.to(device))
              print(f"Epoch: {epoch}, Loss: {loss.item()}")#, Loss_eval : {eval_loss.item()}")
              torch.save(model.state_dict(), '/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters_2.pth')


    return model, loss_list
