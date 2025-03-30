## PyTorch script for Nonlinear Poisson PINN

## Imports necessary for building model
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import StepLR
import numpy as np
import scipy
import itertools
import pandas as pd
import random


##################################################
## Setup for our networks build
##################################################

# We consider Net as our function u(x,t).  We'll provide data and constraints which ensure that the function
# u(x,t) satisfies a variety of PDEs.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sz = 35
        # Define the layers of the neural network
        self.hidden_layer1 = nn.Linear(3, sz)  
        self.hidden_layer2 = nn.Linear(sz, sz)  
        self.hidden_layer3 = nn.Linear(sz, sz)  
        self.hidden_layer4 = nn.Linear(sz, sz)  
        self.hidden_layer5 = nn.Linear(sz, sz)  
        self.hidden_layer6 = nn.Linear(sz, sz)  
        self.hidden_layer7 = nn.Linear(sz, sz) 
        self.output_layer = nn.Linear(sz, 1)

    def forward(self, x, y, z):
        # Concatenate input tensors x and t along the second axis (columns)
        inputs = torch.cat([x, y, z], axis=1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        layer6_out = torch.tanh(self.hidden_layer6(layer5_out))
        layer7_out = torch.tanh(self.hidden_layer7(layer6_out))
        output = self.output_layer(layer7_out)
        return output

# Instantiate the neural network model
net = Net()
# Move the model to the specified device (assuming device is defined elsewhere in your code)
net = net.to(device)
# Define the mean squared error loss function
mse_cost_function = torch.nn.MSELoss()
# Define the AdamW optimizer with a learning rate of 0.01, optimizing the parameters of the neural network
optimizer = torch.optim.AdamW(net.parameters(), lr=0.007) # classic optimizer for gradient descent
# Adding schedulers so that I can alter learning rate mid-run
scheduler = StepLR(optimizer, step_size=1000, gamma=0.95) # LR decay 
scheduler2 = StepLR(optimizer, step_size=2000, gamma=0.95) # alterate LR decay







##################################################
## Setup for our phsyics model
##################################################

#energy distributuion that we will work with
def delta_rho(x, y, z):
    return 5 #placeholder distribution

## PDE as loss function. Solving the partial differential equation using the neural network-generated solution u
def f(x,y,z, net):
    u = net(x,y,z) # the dependent variable u is given by the network based on independent variables x,y,z
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # u.sum() is necissary to turns the u tensor into a scalar
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    

    pde = u_xx + u_yy + u_zz - delta_rho(x,y,z) * u * u * u * u * u
    return pde


# set domain size
lower_bound = 0
upper_bound = 1








##################################################
## Build my training set.
##################################################

# This is for creating the training set points on the boundaries
def bc(number_pts, lower_bound, upper_bound):
    ## I need points for the x=lower boundary
    x_l = lower_bound*np.ones((number_pts,1))
    y_l = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    z_l = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))

    ## I need points for the x=upper boundary
    x_r = upper_bound*np.ones((number_pts,1))
    y_r = y_l
    z_r = z_l

    ## I need points for the y=lower boundary
    x_ll = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    y_ll = lower_bound*np.ones((number_pts,1))
    z_ll = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))

    ## I need points for the y=upper boundary
    x_rr = x_ll
    y_rr = upper_bound*np.ones((number_pts,1))
    z_rr = z_ll

    ## I need points for the z=lower boundary
    x_lll = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    y_lll = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    z_lll = lower_bound*np.ones((number_pts,1))

    ## I need points for the z=upper boundary
    x_rrr = x_lll
    y_rrr = y_lll
    z_rrr = upper_bound*np.ones((number_pts,1))

    return (x_l,y_l,z_l,x_r,y_r,z_r,x_ll,y_ll,z_ll,x_rr,y_rr,z_rr,x_lll,y_lll,z_lll,x_rrr,y_rrr,z_rrr)

# for creating the training set in the interior
def interior(number_pts):
    x_sampling = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    y_sampling = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    z_sampling = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1))
    u_sampling = np.zeros((number_pts,1))

    return x_sampling, y_sampling, z_sampling, u_sampling


# Interior Points
num_pts_interior = 10000
x_sampling, y_sampling, z_sampling, u_sampling = interior(num_pts_interior)
interior_x_sampling = Variable(torch.from_numpy(x_sampling).float(), requires_grad=True).to(device) # turns BC samples into PyTorch tensors with gradient enabled
interior_y_sampling = Variable(torch.from_numpy(y_sampling).float(), requires_grad=True).to(device)
interior_z_sampling = Variable(torch.from_numpy(z_sampling).float(), requires_grad=True).to(device)
interior_all_zeros = Variable(torch.from_numpy(u_sampling).float(), requires_grad=False).to(device) # tensor to be compared to

num_pts_boundary = 300
x_l,y_l,z_l,x_r,y_r,z_r,x_ll,y_ll,z_ll,x_rr,y_rr,z_rr,x_lll,y_lll,z_lll,x_rrr,y_rrr,z_rrr = bc(num_pts_boundary, lower_bound, upper_bound)

x_l_sampling = Variable(torch.from_numpy(x_l).float(), requires_grad=True).to(device)
y_l_sampling = Variable(torch.from_numpy(y_l).float(), requires_grad=True).to(device)
z_l_sampling = Variable(torch.from_numpy(z_l).float(), requires_grad=True).to(device)

x_r_sampling = Variable(torch.from_numpy(x_r).float(), requires_grad=True).to(device)
y_r_sampling = Variable(torch.from_numpy(y_r).float(), requires_grad=True).to(device)
z_r_sampling = Variable(torch.from_numpy(z_r).float(), requires_grad=True).to(device)

x_ll_sampling = Variable(torch.from_numpy(x_ll).float(), requires_grad=True).to(device)
y_ll_sampling = Variable(torch.from_numpy(y_ll).float(), requires_grad=True).to(device)
z_ll_sampling = Variable(torch.from_numpy(z_ll).float(), requires_grad=True).to(device)

x_rr_sampling = Variable(torch.from_numpy(x_rr).float(), requires_grad=True).to(device)
y_rr_sampling = Variable(torch.from_numpy(y_rr).float(), requires_grad=True).to(device)
z_rr_sampling = Variable(torch.from_numpy(z_rr).float(), requires_grad=True).to(device)

x_lll_sampling = Variable(torch.from_numpy(x_lll).float(), requires_grad=True).to(device)
y_lll_sampling = Variable(torch.from_numpy(y_lll).float(), requires_grad=True).to(device)
z_lll_sampling = Variable(torch.from_numpy(z_lll).float(), requires_grad=True).to(device)

x_rrr_sampling = Variable(torch.from_numpy(x_rrr).float(), requires_grad=True).to(device)
y_rrr_sampling = Variable(torch.from_numpy(y_rrr).float(), requires_grad=True).to(device)
z_rrr_sampling = Variable(torch.from_numpy(z_rrr).float(), requires_grad=True).to(device)


##################################################
## Starting code for Training / Fitting
##################################################

iterations = 200

for epoch in range(iterations):
    # optimizer.zero_grad() should be indented properly
    optimizer.zero_grad()  # to make the gradients zero
    
    # Loss based on PDE
    f_out = f(interior_x_sampling, interior_y_sampling, interior_z_sampling, net)  # output of f(x,t)
    mse_interior = mse_cost_function(f_out, interior_all_zeros)

    l_out = f(x_l_sampling, y_l_sampling, z_l_sampling, net)
    r_out = f(x_r_sampling, y_r_sampling, z_r_sampling, net)
    ll_out = f(x_ll_sampling, y_ll_sampling, z_ll_sampling, net)
    rr_out = f(x_rr_sampling, y_rr_sampling, z_rr_sampling, net)
    lll_out = f(x_lll_sampling, y_lll_sampling, z_lll_sampling, net)
    rrr_out = f(x_rrr_sampling, y_rrr_sampling, z_rrr_sampling, net)
    mse_boundary = mse_cost_function(l_out, r_out) + mse_cost_function(ll_out, rr_out) + mse_cost_function(lll_out, rrr_out)
    
    loss = mse_interior + 0.1 * mse_boundary


    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    
    with torch.autograd.no_grad():
        if epoch % 20 == 0:
            print(epoch,"Training Loss:",loss.item(), ", Current learning rate:", scheduler.get_last_lr()[0], ", Current number of training points:",(x_l_sampling.size(0) + x_ll_sampling.size(0) + x_lll_sampling.size(0) + x_r_sampling.size(0) + x_rr_sampling.size(0) + x_rrr_sampling.size(0) + interior_x_sampling.size(0)), " Current VRAM allocated:", torch.cuda.memory_allocated(), "Current VRAM reserved:", torch.cuda.memory_reserved())

## Save my new model
torch.save(net.state_dict(), './power_five.pt')



