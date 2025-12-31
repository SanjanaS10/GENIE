# standard system modules
import os, sys

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flowde_pinn import gendata 
from flowde_pinn import flow_de
from flowde_pinn import networks_3d

base_path = "/pscratch/sd/s/sijilj/PINN/python_files/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device: {str(DEVICE):4s}')

# Loading the data :

X0, _ = gendata.load_or_generate_gmm_2d(path=base_path+"X03d.pth", N=50000, K=20, device=DEVICE)

N = 5000
M = 8000
print(X0.shape)

## Numerically solving the flow equation
x0 = flow_de.get_target_sample(X0, M).to(DEVICE)
flow = flow_de.FlowDE(x0,T= 250, savepath=True)

z = flow_de.get_normal_sample(X0[:N]).to(DEVICE)
y,times = flow(z)

print(len(y))

N_col = 1000  # number of collocation points
z_train = flow_de.get_normal_sample(X0[:N_col])

#z_train = np.linspace(-5,5,N_col)

#np.random.shuffle(z_train)

z_train = torch.Tensor(z_train[:, None,None])

t_data = np.array(times)
z_data = np.array(z.cpu())
x_data = np.array(y)

######### Training the Network


networks_3d.set_seed(42)


# Instantiating the model

model = networks_3d.PhysicsNN(x0,z).to(DEVICE)
model.load_state_dict(torch.load('/pscratch/sd/s/sijilj/PINN/RFDE_model_parameters_3D.pth'))


#model, loss_list = train(model,z_train, t_data, z_data, x_data, ramp_steps=8, initial_z_frac=1, lr_target=1, lr_warmup_factor=0.1 )
