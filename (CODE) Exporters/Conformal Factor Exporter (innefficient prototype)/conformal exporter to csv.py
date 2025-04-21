import torch
import torch.nn as nn
import math
import numpy as np
import csv

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sz = 400
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
net = net.to(device)

checkpoint = torch.load('pre-train_checkpoint_epoch=990000.pth', map_location=torch.device(device), weights_only=True) # load the previous training
net.load_state_dict(checkpoint['net_state_dict']) # load the model

# Physics paramters
H = math.sqrt(8*np.pi/3) # Physics quantity called the hubble parameter
lower_bound = -2.5*np.pi/H # Box size is relative to H
upper_bound = 2.5*np.pi/H
sigma = 1.1/H # Sigma paramter for the Gaussian distribution


file1 = open('model_export.csv','w') # Open csv file for data output
writer1 = csv.writer(file1)

with torch.no_grad():
    for xxx in range(0,128): # loop over the field
        for yyy in range(0,128):
            for zzz in range(0,128):
                x = torch.tensor([[lower_bound + xxx * (upper_bound - lower_bound) / 127]], dtype=torch.float64).to(device)
                y = torch.tensor([[lower_bound + yyy * (upper_bound - lower_bound) / 127]], dtype=torch.float64).to(device)
                z = torch.tensor([[lower_bound + zzz * (upper_bound - lower_bound) / 127]], dtype=torch.float64).to(device)
                    
                output = net(x, y, z)
                
                writer1.writerow([output])

print("Confromal export complete.")
