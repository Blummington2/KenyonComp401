## PyTorch script for Nonlinear Poisson PINN

## Imports necessary for building model
import torch # PyTorch package for machine learning structures
import torch.nn as nn # Neural network architecture
from torch.autograd import Variable  # Variables for gradient calculations in neural networks
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Enables GPU-accelerated computation with CUDA if available, otherwise computes on CPU
from torch.optim.lr_scheduler import StepLR # For variable learning rate in the MLA
import numpy as np # Numpy for data structure manipulation
import matplotlib.pyplot as plt # For graphical visualizations
import csv # For data output into a csv file
import math # For certain math operations


# Set the default data type to float64
torch.set_default_dtype(torch.float64) # hasn't worked so far... so we're working with float32








##################################################
## Setup for network build
##################################################

# We consider Net as our function u(x,t).  We'll provide constraints which ensure that the function solves the problem at hand:
# Nonlinear Poisson equation with energy-field-source, delta-rho, and periodic boundary conditions.
class Net(nn.Module): # Net inherits the neural network behavior of nn.Module
    def __init__(self): # Class constructor with no additional aarguments
        super(Net, self).__init__() # Calls the nn.Module constructor, giving our class the properties of nn.Module, including gradient computation and more...
        sz = 456 # Set neurons per layer
        # Define the layers of the neural network
        self.hidden_layer1 = nn.Linear(3, sz)  
        self.hidden_layer2 = nn.Linear(sz, sz)  
        self.hidden_layer3 = nn.Linear(sz, sz)  
        self.hidden_layer4 = nn.Linear(sz, sz)  
        self.hidden_layer5 = nn.Linear(sz, sz)  
        self.hidden_layer6 = nn.Linear(sz, sz)  
        self.hidden_layer7 = nn.Linear(sz, sz) 
        self.output_layer = nn.Linear(sz, 1)

    def forward(self, x, y, z): # Each layer passes to the next with the tanh activation function
        # Concatenate input tensors x, y, and z along the second axis (columns)
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

# Construct the neural network
net = Net()
# Move the model to the specified device (assuming device is defined elsewhere in your code)
net = net.to(device)
# Define the mean squared error loss function
mse_cost_function = torch.nn.MSELoss()
# Define the AdamW optimizer. Excecutes the gradient-descent-like optimization of the parameters of the neural network (Classic optimizer)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.004) 
# Adding schedulers so that I can alter learning rate mid-run
scheduler = StepLR(optimizer, step_size=1000, gamma=0.98) # LR decay 
scheduler2 = StepLR(optimizer, step_size=2000, gamma=0.99) # alterate LR decay







##################################################
## Setup for phsyics model
##################################################

# Energy distributuion that we will work with. Guassian distribution
def delta_rho(x, y, z): 
    r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z)) # Calculate radius based on spatial position
    return (1/(sigma*math.sqrt(2*torch.pi)))*torch.exp(-0.5*r*r/sigma/sigma) - 0.0056776 # Gaussian distribution

# PDE as loss function. Solving the partial differential equation using the neural-network-generated solution u
def f(x,y,z, net): 
    u = net(x,y,z) # The dependent variable u is given by the network based on independent variables x,y,z
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # u.sum() is necissary to turns the u tensor into a scalar
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    pde = (u_xx + u_yy + u_zz) / u / u / u / u / u - 2 * math.pi * delta_rho(x,y,z) # Loss at any given point is defined in terms of the differential equation
    return pde

def f_linear(x,y,z, net): # Same as f(), but without nonlinear part
    u = net(x,y,z) 
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] 
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    u_z = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z.sum(), z, create_graph=True)[0]

    pde = (u_xx + u_yy + u_zz) - 2 * math.pi * delta_rho(x,y,z)





# Set domain size and other quantities. Quantities based on those used by Tom Giblin and Amanda Miller in their physics research
H = math.sqrt(8*np.pi/3) # Physics quantity called the hubble parameter
lower_bound = -2.5*np.pi/H # Box size is relative to H
upper_bound = 2.5*np.pi/H
sigma = 1.1/H # Sigma paramter for the Gaussian distribution
















##################################################
## Visualization functions
##################################################

def visualize_function(lb= lower_bound, ub= upper_bound, num_slices=5, domain_size=100, device='cpu',epoch = 100): # Graphs a few slices of the full box
    # Create the domain grid
    x = np.linspace(lb, ub, domain_size) # Creates sampling accross the x-coordinate grid
    y = np.linspace(lb, ub, domain_size) # Creates sampling accross the y-coordinate grid
    z_slices = np.linspace(lb, ub, num_slices)  # Slicing along the z-axis

    X, Y = np.meshgrid(x, y)  # Create grid for X and Y 
    x = np.ravel(X).reshape(-1, 1) # Flattens X from the meshgrid into a flat, 1D array. Then reshapes to [[],[],[],[]] format
    y = np.ravel(Y).reshape(-1, 1) # Flattens Y from the meshgrid into a flat, 1D array. Then reshapes to [[],[],[],[]] format
    
    sample_x = Variable(torch.from_numpy(x).double(), requires_grad=True).to(device) # Converts to a torch tensor and then to a torch variable tensor
    sample_y = Variable(torch.from_numpy(y).double(), requires_grad=True).to(device) # Converts to a torch tensor and then to a torch variable tensor

    # Initialize a list to collect u values across slices
    u_values = [] # Empty array

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8)) # Creates visualization
    ax = fig.add_subplot(projection='3d') # Creates plot within the visualization

    # Collect u values to determine the global min/max
    for z in z_slices:
        zz = 0.0 * x + z # Array of z-values for constant z
        zzz = np.ravel(zz).reshape(-1, 1) # flattens into 1D, then turns into [[],[],[],[]] format
        sample_z = Variable(torch.from_numpy(zzz).double(), requires_grad=True).to(device) # Converts to a torch tensor and then to a torch variable tensor

        
        Z = net(sample_x,sample_y,sample_z) # Get field values
        u = Z.data.cpu().numpy().reshape(X.shape) # Collects fields values, moves to CPU, converts to numpy, converts shape to match the shape of X and Y
        
        u_values.append(u)  # Collect u values together for later analysis

    # Now that we have all u values, determine the min/max across all slices
    u_values = np.array(u_values)  # This converts from a list of numpy arrays to a numpy array of numpy arrays (shape: [num_slices, X.shape])
    vmin, vmax = np.min(u_values), np.max(u_values) # Calculates minimum and maximum field value within the slices

    # Plot the slices with the correct vmin/vmax
    for z, u in zip(z_slices, u_values): # Iterate over each z-slice, extracting the corresponding field-value array and naming it u
        ax.contourf(X, Y, u, zdir='z', offset=z, cmap=plt.cm.plasma , alpha=0.8, levels=50, vmin=vmin, vmax=vmax) # Plots u, using X and Y as coordinate values

    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.plasma , norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(m, ax=ax, shrink=0.4, aspect=10) 

    # Set labels and titles
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlim(lb, ub)
    ax.set_zlabel('Function Value')
    ax.set_title(r'Function Value on the Grid')
    ax.view_init(elev=15, azim=-4, roll=0) # Set viewing perspective

    # Save the plot
    plt.savefig('poisson_func_'+str(epoch)+'.png')
    plt.close()



def visualize_error(lb= lower_bound, ub= upper_bound, num_slices=5, domain_size=50, device='cpu', epoch = 0):
    # Create the domain grid
    x = np.linspace(lb, ub, domain_size)
    y = np.linspace(lb, ub, domain_size)
    z_slices = np.linspace(lb, ub, num_slices)  # Slicing along the z-axis

    X, Y = np.meshgrid(x, y)  # Create grid for X and Y
    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    
    sample_x = Variable(torch.from_numpy(x).double(), requires_grad=True).to(device)
    sample_y = Variable(torch.from_numpy(y).double(), requires_grad=True).to(device)

    # Initialize a list to collect u values across slices
    u_values = []

    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # First pass: collect u values to determine the global min/max
    for z in z_slices:
        zz = 0.0 * x + z
        zzz = np.ravel(zz).reshape(-1, 1)
        sample_z = Variable(torch.from_numpy(zzz).double(), requires_grad=True).to(device)
        
        F = f(sample_x, sample_y, sample_z, net) # get field error values
        u = F.data.cpu().numpy().reshape(X.shape)
        
        # Collect u values for later analysis
        u_values.append(u)

    # Now that we have all u values, determine the min/max across all slices
    u_values = np.array(u_values)  # This keeps the list of arrays (shape: [num_slices, X.shape])
    vmin, vmax = np.min(u_values), np.max(u_values)

    # Second pass: plot the slices with the correct vmin/vmax
    for z, u in zip(z_slices, u_values):
        ax.contourf(X, Y, u, zdir='z', offset=z, cmap=plt.cm.plasma, alpha=0.8, levels=50, vmin=vmin, vmax=vmax)

    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(m, ax=ax, shrink=0.4, aspect=10)

    # Set labels and titles
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlim(lb, ub)
    ax.set_zlabel('Function Value')
    ax.set_title(r'PDE Error near the source')
    ax.view_init(elev=15, azim=-40, roll=0)

    # Save and show the plot
    plt.savefig('poisson_err_'+str(epoch)+'.png')
    plt.close()



def visualize_x0_slice(lb=lower_bound,ub=upper_bound, domain_size=100,device='cpu', epoch=0):
    # Create the domain grid
    y = np.linspace(lb, ub, domain_size)  # Slicing along the y-axis
    z = np.linspace(lb, ub, domain_size)  # Slicing along the z-axis
   
    Y, Z = np.meshgrid(y, z)  # Create grid for X and Y
    y = np.ravel(Y).reshape(-1,1)
    z = np.ravel(Z).reshape(-1,1)
    sample_y = Variable(torch.from_numpy(y).double(), requires_grad=False).to(device)
    sample_z = Variable(torch.from_numpy(z).double(), requires_grad=False).to(device)
    
    xx = 0.0*y # x=0 is constant because we are taking a 2D slice through the origin
    sample_x = Variable(torch.from_numpy(xx).double(), requires_grad=False).to(device)
    
    U = net(sample_x,sample_y,sample_z)
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
    x_l = lower_bound*np.ones((number_pts,1)) # Creates array [[lb],[lb],[lb],...,[lb]]
    y_l = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1)) # Random sampling within domain values
    z_l = np.random.uniform(low=lower_bound, high=upper_bound, size=(number_pts,1)) # Random sampling within domain values

    ## I need points for the x=upper boundary
    x_r = upper_bound*np.ones((number_pts,1)) # Creates array [[ub],[ub],[ub],...,[ub]]
    y_r = y_l # For periodic boundary conditions. We need to compare the same (y,z) coordinates on both sides of the grid. 
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



# Consider using np.concatenate rather than np.append for efficieny (the gains would be small). Use ChatGPT for help.
# For creating the training set in the interior
def interior(number_pts): # Creates sampling with higher frequency towards the center of the box where we expect complicated dynamics
    x_sampling = np.random.uniform(low=lower_bound, high=upper_bound, size=((int)(number_pts/2),1))
    y_sampling = np.random.uniform(low=lower_bound, high=upper_bound, size=((int)(number_pts/2),1))
    z_sampling = np.random.uniform(low=lower_bound, high=upper_bound, size=((int)(number_pts/2),1)) # Radnom sampling accross the grid

    x_sampling = np.append(x_sampling, np.random.normal(loc=0, scale=sigma, size=((int)(number_pts/3),1)))
    y_sampling = np.append(y_sampling, np.random.normal(loc=0, scale=sigma, size=((int)(number_pts/3),1)))
    z_sampling = np.append(z_sampling, np.random.normal(loc=0, scale=sigma, size=((int)(number_pts/3),1))) # Random sampling with Gaussian frequency distribution

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/10, high=upper_bound/10, size=((int)(number_pts/12),1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/10, high=upper_bound/10, size=((int)(number_pts/12),1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/10, high=upper_bound/10, size=((int)(number_pts/12),1))) # Random sampling accross part of the grid

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/100, high=upper_bound/100, size=((int)(number_pts/24),1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/100, high=upper_bound/100, size=((int)(number_pts/24),1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/100, high=upper_bound/100, size=((int)(number_pts/24),1))) # Random sampling accross part of the grid, increasingly close to center

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/500, high=upper_bound/500, size=((int)(number_pts/24 -30),1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/500, high=upper_bound/500, size=((int)(number_pts/24 -30),1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/500, high=upper_bound/500, size=((int)(number_pts/24 -30),1)))

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/5000, high=upper_bound/5000, size=(15,1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/5000, high=upper_bound/5000, size=(15,1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/5000, high=upper_bound/5000, size=(15,1)))

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/50000, high=upper_bound/50000, size=(6,1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/50000, high=upper_bound/50000, size=(6,1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/50000, high=upper_bound/50000, size=(6,1)))

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/500000, high=upper_bound/500000, size=(4,1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/500000, high=upper_bound/500000, size=(4,1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/500000, high=upper_bound/500000, size=(4,1)))

    x_sampling = np.append(x_sampling, np.random.uniform(low=lower_bound/5000000, high=upper_bound/5000000, size=(2,1)))
    y_sampling = np.append(y_sampling, np.random.uniform(low=lower_bound/5000000, high=upper_bound/5000000, size=(2,1)))
    z_sampling = np.append(z_sampling, np.random.uniform(low=lower_bound/5000000, high=upper_bound/5000000, size=(2,1)))

    for XXX in range(3):
        x_sampling = np.append(x_sampling, [[0]])
        y_sampling = np.append(y_sampling, [[0]])
        z_sampling = np.append(z_sampling, [[0]]) # 3 Sample points at the origin

    u_sampling = np.zeros((len(x_sampling),1))


    return x_sampling, y_sampling, z_sampling, u_sampling


# Interior points
num_pts_interior = 30000
x_sampling, y_sampling, z_sampling, u_sampling = interior(num_pts_interior)
interior_x_sampling = Variable(torch.from_numpy(x_sampling).double(), requires_grad=True).to(device).unsqueeze(1) # turns BC samples into PyTorch tensors with gradient enabled
interior_y_sampling = Variable(torch.from_numpy(y_sampling).double(), requires_grad=True).to(device).unsqueeze(1) # unsqueeze fixes dimesnions
interior_z_sampling = Variable(torch.from_numpy(z_sampling).double(), requires_grad=True).to(device).unsqueeze(1)
interior_all_zeros = Variable(torch.from_numpy(u_sampling).double(), requires_grad=False).to(device) # tensor to be compared to

# Boundary points
num_pts_boundary = 800
x_l,y_l,z_l,x_r,y_r,z_r,x_ll,y_ll,z_ll,x_rr,y_rr,z_rr,x_lll,y_lll,z_lll,x_rrr,y_rrr,z_rrr = bc(num_pts_boundary, lower_bound, upper_bound)

x_l_sampling = Variable(torch.from_numpy(x_l).double(), requires_grad=True).to(device)
y_l_sampling = Variable(torch.from_numpy(y_l).double(), requires_grad=True).to(device)
z_l_sampling = Variable(torch.from_numpy(z_l).double(), requires_grad=True).to(device)

x_r_sampling = Variable(torch.from_numpy(x_r).double(), requires_grad=True).to(device)
y_r_sampling = Variable(torch.from_numpy(y_r).double(), requires_grad=True).to(device)
z_r_sampling = Variable(torch.from_numpy(z_r).double(), requires_grad=True).to(device)

x_ll_sampling = Variable(torch.from_numpy(x_ll).double(), requires_grad=True).to(device)
y_ll_sampling = Variable(torch.from_numpy(y_ll).double(), requires_grad=True).to(device)
z_ll_sampling = Variable(torch.from_numpy(z_ll).double(), requires_grad=True).to(device)

x_rr_sampling = Variable(torch.from_numpy(x_rr).double(), requires_grad=True).to(device)
y_rr_sampling = Variable(torch.from_numpy(y_rr).double(), requires_grad=True).to(device)
z_rr_sampling = Variable(torch.from_numpy(z_rr).double(), requires_grad=True).to(device)

x_lll_sampling = Variable(torch.from_numpy(x_lll).double(), requires_grad=True).to(device)
y_lll_sampling = Variable(torch.from_numpy(y_lll).double(), requires_grad=True).to(device)
z_lll_sampling = Variable(torch.from_numpy(z_lll).double(), requires_grad=True).to(device)

x_rrr_sampling = Variable(torch.from_numpy(x_rrr).double(), requires_grad=True).to(device)
y_rrr_sampling = Variable(torch.from_numpy(y_rrr).double(), requires_grad=True).to(device)
z_rrr_sampling = Variable(torch.from_numpy(z_rrr).double(), requires_grad=True).to(device)





print("Data-type = ", interior_x_sampling.dtype)

##################################################
## Starting code for Training / Fitting
##################################################

file1 = open('epoch_residual.csv','w') # Open csv file for data output
writer1 = csv.writer(file1) # Writes to csv file

iterations = 1200000 # Number of loops in training



for epoch in range(iterations): # 'Epoch' keeps track of current training iteration

    optimizer.zero_grad()  # to make the gradients zero. Clear gradients

    if epoch % 10000 == 0: # Every 10000 epochs...
        visualize_error(lb= lower_bound, ub= upper_bound, num_slices=5, domain_size=100, device=device, epoch = epoch)
        with torch.autograd.no_grad():
            visualize_function(lb=lower_bound, ub=upper_bound, num_slices=5, domain_size=100, device=device, epoch=epoch)
            visualize_x0_slice(lb=0.1*lower_bound,ub=0.1*upper_bound, domain_size=100,device=device, epoch=epoch)
            visualize_x0_slice(lb=lower_bound,ub=upper_bound, domain_size=100,device=device, epoch=epoch)
            torch.save({'net_state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, 'pre-pre-train_checkpoint_epoch=' + str(epoch) + '.pth') # Saves current network and optimizer state so that training can be resumed from any point
    
    # Loss based on PDE
    net_out = net(interior_x_sampling, interior_y_sampling, interior_z_sampling)  # Output of net
    sphere = 400 * delta_rho(interior_x_sampling, interior_y_sampling, interior_z_sampling) # the distribution that we want to start with
    mse_interior = mse_cost_function(net_out, sphere) # Automatically calculates interior loss with gradient disabled

    """
    l_out = net(x_l_sampling, y_l_sampling, z_l_sampling) # Field values on borders
    r_out = net(x_r_sampling, y_r_sampling, z_r_sampling)
    ll_out = net(x_ll_sampling, y_ll_sampling, z_ll_sampling)
    rr_out = net(x_rr_sampling, y_rr_sampling, z_rr_sampling)
    lll_out = net(x_lll_sampling, y_lll_sampling, z_lll_sampling)
    rrr_out = net(x_rrr_sampling, y_rrr_sampling, z_rrr_sampling)
    mse_boundary = mse_cost_function(l_out, r_out) + mse_cost_function(ll_out, rr_out) + mse_cost_function(lll_out, rrr_out) # Periodic boundary conditions... Loss higher if opposite boundaries don't match
    """
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
    loss = mse_interior # + 0.333 * mse_boundary # Total loss


    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative. Gradient Descent.

    if epoch > 30000 and epoch < 200000:
        scheduler.step() # Adjust learning rate gradually
    if epoch > 400000 and epoch < 1000000:
        scheduler2.step() # Slower adjusting of learning rate gradually
    
    if epoch % 100 == 0:
        with torch.autograd.no_grad(): # Data output to terminal and also to the csv file. Every 100 epochs.
            writer1.writerow(["Epoch:", epoch ,"Training Loss:",loss.item(), ", Learning rate:", scheduler.get_last_lr()[0], ", Number of training points (boundary):",(x_l_sampling.size(0) + x_ll_sampling.size(0) + x_lll_sampling.size(0) + x_r_sampling.size(0) + x_rr_sampling.size(0) + x_rrr_sampling.size(0)), ", Number of training points (interior):", interior_x_sampling.size(0), ", VRAM allocated:", torch.cuda.memory_allocated(), ", VRAM reserved:", torch.cuda.memory_reserved()])
            print("Epoch:", epoch, "Training Loss:",loss.item(), " Learning rate:", scheduler.get_last_lr()[0], " Number of training points (boundary):",(x_l_sampling.size(0) + x_ll_sampling.size(0) + x_lll_sampling.size(0) + x_r_sampling.size(0) + x_rr_sampling.size(0) + x_rrr_sampling.size(0)), " Number of training points (interior):", interior_x_sampling.size(0), " VRAM allocated:", torch.cuda.memory_allocated(), " VRAM reserved:", torch.cuda.memory_reserved())

## Save my new model
torch.save(net.state_dict(), './pre-pre-train.pt')



