## PyTorch script for Nonlinear Poisson PINN

## Imports necessary for building model
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import pandas as pd
import random
import csv
import math

# Set the default data type to float64
# torch.set_default_dtype(torch.float64) # hasn't worked so far... so we're working with float32

##################################################
## Setup for our networks build
##################################################

# We consider Net as our function u(x,t).  We'll provide data and constraints which ensure that the function
# u(x,t) satisfies a variety of PDEs.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sz = 100
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
optimizer = torch.optim.AdamW(net.parameters(), lr=0.00008) # classic optimizer for gradient descent
# Adding schedulers so that I can alter learning rate mid-run
scheduler = StepLR(optimizer, step_size=1000, gamma=0.95) # LR decay 
scheduler2 = StepLR(optimizer, step_size=2000, gamma=0.95) # alterate LR decay







##################################################
## Setup for our phsyics model
##################################################

#energy distributuion that we will work with
def delta_rho(x, y, z):
    r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))
    return (1/(sigma*math.sqrt(2*torch.pi)))*torch.exp(-0.5*r*r/sigma)

## PDE as loss function. Solving the partial differential equation using the neural network-generated solution u
def f(x,y,z, net):
    u = net(x,y,z) # the dependent variable u is given by the network based on independent variables x,y,z
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # u.sum() is necissary to turns the u tensor into a scalar
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    pde = (u_xx + u_yy + u_zz) / u / u / u / u / u - 2 * math.pi * delta_rho(x,y,z)
    return pde

def f_linear(x,y,z, net): # removing nonlinear part
    u = net(x,y,z) # the dependent variable u is given by the network based on independent variables x,y,z
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # u.sum() is necissary to turns the u tensor into a scalar
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    pde = (u_xx + u_yy + u_zz) - 2 * math.pi * delta_rho(x,y,z)





# set domain size
lower_bound = -10
upper_bound = 10
sigma = 1
















##################################################
## Visualization functions
##################################################

def visualize_function(lb= lower_bound, ub= upper_bound, num_slices=5, domain_size=100, device='cpu',epoch = 100):
    # Create the domain grid
    x = np.linspace(lb, ub, domain_size)
    y = np.linspace(lb, ub, domain_size)
    z_slices = np.linspace(lb, ub, num_slices)  # Slicing along the z-axis

    X, Y = np.meshgrid(x, y)  # Create grid for X and Y
    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    
    sample_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    sample_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)

    # Initialize a list to collect u values across slices
    u_values = []

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.cm.plasma

    # First pass: collect u values to determine the global min/max
    for z in z_slices:
        zz = 0.0 * x + z
        zzz = np.ravel(zz).reshape(-1, 1)
        sample_z = Variable(torch.from_numpy(zzz).float(), requires_grad=True).to(device)
        
        # Compute F and J (assuming net, f, and jay_per are defined)
        Z = net(sample_x,sample_y,sample_z)
        u = Z.data.cpu().numpy().reshape(X.shape)
        
        # Collect u values for later analysis
        u_values.append(u)

    # Now that we have all u values, determine the min/max across all slices
    u_values = np.array(u_values)  # This keeps the list of arrays (shape: [num_slices, X.shape])
    vmin, vmax = np.min(u_values), np.max(u_values)

    # Second pass: plot the slices with the correct vmin/vmax
    for z, u in zip(z_slices, u_values):
        ax.contourf(X, Y, u, zdir='z', offset=z, cmap=cmap, alpha=0.8, levels=50, vmin=vmin, vmax=vmax)

    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(m, ax=ax, shrink=0.4, aspect=10)

    # Set labels and titles
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlim(lb, ub)
    ax.set_zlabel('Function Value')
    ax.set_title(r'Function Value near the source')
    ax.view_init(elev=15, azim=-45, roll=0)

    # Save and show the plot
    plt.savefig('Galileon_func_'+str(epoch)+'.png')
    plt.close()



def visualize_error(lb= lower_bound, ub= upper_bound, num_slices=5, domain_size=50, device='cpu', epoch = 0):
    # Create the domain grid
    x = np.linspace(lb, ub, domain_size)
    y = np.linspace(lb, ub, domain_size)
    z_slices = np.linspace(lb, ub, num_slices)  # Slicing along the z-axis

    X, Y = np.meshgrid(x, y)  # Create grid for X and Y
    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)

    # Initialize a list to collect u values across slices
    u_values = []

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.cm.plasma

    # First pass: collect u values to determine the global min/max
    for z in z_slices:
        zz = 0.0 * x + z
        zzz = np.ravel(zz).reshape(-1, 1)
        pt_z = Variable(torch.from_numpy(zzz).float(), requires_grad=True).to(device)
        
        # Compute F and J (assuming net, f, and jay_per are defined)
        F = f(pt_x, pt_y, pt_z, net)
        u = F.data.cpu().numpy().reshape(X.shape)
        
        # Collect u values for later analysis
        u_values.append(u)

    # Now that we have all u values, determine the min/max across all slices
    u_values = np.array(u_values)  # This keeps the list of arrays (shape: [num_slices, X.shape])
    vmin, vmax = np.min(u_values), np.max(u_values)

    # Second pass: plot the slices with the correct vmin/vmax
    for z, u in zip(z_slices, u_values):
        ax.contourf(X, Y, u, zdir='z', offset=z, cmap=cmap, alpha=0.8, levels=50, vmin=vmin, vmax=vmax)

    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(m, ax=ax, shrink=0.4, aspect=10)

    # Set labels and titles
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlim(lb, ub)
    ax.set_zlabel('Function Value')
    ax.set_title(r'PDE Error near the source')
    ax.view_init(elev=15, azim=-45, roll=0)

    # Save and show the plot
    plt.savefig('Galileon_err_'+str(epoch)+'.png')
    plt.close()



def visualize_x0_slice(lb=lower_bound,ub=upper_bound, domain_size=100,device='cpu', epoch=0):
    # Create the domain grid
    y = np.linspace(lb, ub, domain_size)
    z = np.linspace(lb, ub, domain_size)  # Slicing along the z-axis
   
    Y, Z = np.meshgrid(y, z)  # Create grid for X and Y
    y = np.ravel(Y).reshape(-1,1)
    z = np.ravel(Z).reshape(-1,1)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False).to(device)
    pt_z = Variable(torch.from_numpy(z).float(), requires_grad=False).to(device)
    
    xx = 0.0*y
    pt_x = Variable(torch.from_numpy(xx).float(), requires_grad=False).to(device)
    
    U = net(pt_x,pt_y,pt_z)
    u=U.data.cpu().numpy().reshape(Y.shape)
    plt.figure(figsize=(8, 6))
    #cmap = cm.plasma
    contour = plt.contour(Y, Z, u, levels=50, cmap='plasma')  # Contour levels can be adjusted
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title(r'Function value at x=0')
    plt.colorbar(contour)

    if ub > 0.5 * upper_bound:
        plt.savefig('Slice_x=0_'+ str(epoch) + '.png')
    else:
        plt.savefig('Slice_x=0_closeup_'+ str(epoch) + '.png')
    plt.close()




















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

num_pts_boundary = 800
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

file1 = open('epoch_residual.csv','w')
writer1 = csv.writer(file1)

iterations = 20000

net.load_state_dict(torch.load('./poisson_pre-train_epoch=9000.pt'))

for epoch in range(iterations):
    # optimizer.zero_grad() should be indented properly
    optimizer.zero_grad()  # to make the gradients zero


    if epoch % 1000 == 0:
        visualize_error(lb= lower_bound, ub= upper_bound, num_slices=5, domain_size=100, device=device, epoch = epoch)
        with torch.autograd.no_grad():
            visualize_function(lb=lower_bound, ub=upper_bound, num_slices=5, domain_size=100, device=device, epoch=epoch)
            visualize_x0_slice(lb=lower_bound,ub=upper_bound, domain_size=100,device=device, epoch=epoch)
    
    # Loss based on PDE
    f_out = f(interior_x_sampling, interior_y_sampling, interior_z_sampling, net)  # output of f(x,t)
    mse_interior = mse_cost_function(f_out, interior_all_zeros)

    l_out = net(x_l_sampling, y_l_sampling, z_l_sampling)
    r_out = net(x_r_sampling, y_r_sampling, z_r_sampling)
    ll_out = net(x_ll_sampling, y_ll_sampling, z_ll_sampling)
    rr_out = net(x_rr_sampling, y_rr_sampling, z_rr_sampling)
    lll_out = net(x_lll_sampling, y_lll_sampling, z_lll_sampling)
    rrr_out = net(x_rrr_sampling, y_rrr_sampling, z_rrr_sampling)
    mse_boundary = mse_cost_function(l_out, r_out) + mse_cost_function(ll_out, rr_out) + mse_cost_function(lll_out, rrr_out)

    # optional for second BC: 
    """
    l_f = f(x_l_sampling, y_l_sampling, z_l_sampling)
    r_f = f(x_r_sampling, y_r_sampling, z_r_sampling)
    ll_f = f(x_ll_sampling, y_ll_sampling, z_ll_sampling)
    rr_f = f(x_rr_sampling, y_rr_sampling, z_rr_sampling)
    lll_f = f(x_lll_sampling, y_lll_sampling, z_lll_sampling)
    rrr_f = f(x_rrr_sampling, y_rrr_sampling, z_rrr_sampling)
    mse_boundary2 = mse_cost_function(l_f, r_f) + mse_cost_function(ll_f, rr_f) + mse_cost_function(lll_f, rrr_f)
    """
    loss = mse_interior + 0.333 * mse_boundary


    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    
    if epoch % 100 == 0:
        with torch.autograd.no_grad():
            writer1.writerow( [epoch, loss.item()])
            print(epoch,"Training Loss:",loss.item(), ", Learning rate:", scheduler.get_last_lr()[0], ", Number of training points (boundary):",(x_l_sampling.size(0) + x_ll_sampling.size(0) + x_lll_sampling.size(0) + x_r_sampling.size(0) + x_rr_sampling.size(0) + x_rrr_sampling.size(0)), ", Number of training points (interior):", interior_x_sampling.size(0), ", VRAM allocated:", torch.cuda.memory_allocated(), ", VRAM reserved:", torch.cuda.memory_reserved())

## Save my new model
torch.save(net.state_dict(), './power_five.pt')



