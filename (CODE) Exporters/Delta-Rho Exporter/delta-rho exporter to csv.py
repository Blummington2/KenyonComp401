import math
import numpy as np
import csv


def delta_rho(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    return (1/(sigma*math.sqrt(2*math.pi)))*math.exp(-0.5*r*r/sigma)

# Physics paramters
H = math.sqrt(8*np.pi/3) # Physics quantity called the hubble parameter
lower_bound = -2.5*np.pi/H # Box size is relative to H
upper_bound = 2.5*np.pi/H
sigma = 1.1/H # Sigma paramter for the Gaussian distribution


file1 = open('delta-rho_model_export.csv','w') # Open csv file for data output
writer1 = csv.writer(file1)

for xxx in range(0,128): # loop over the field
    for yyy in range(0,128):
        for zzz in range(0,128):
            x = lower_bound + xxx * (upper_bound - lower_bound) / 127
            y = lower_bound + yyy * (upper_bound - lower_bound) / 127
            z = lower_bound + zzz * (upper_bound - lower_bound) / 127
                    
            output = delta_rho(x,y,z)
                
            writer1.writerow([output])

print("Delta-rho export complete.")