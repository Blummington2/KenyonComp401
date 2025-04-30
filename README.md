# Physics Informed Neural Network Solutions to Nonlinear Poisson Equations

**Table of Contents**

1. Notes for the Reader/Viewer/Researcher
2. Overview of Project
3. Code/Training Versions Guide
4. For the Casual Reader
5. Notes on Techniques Used
6. Conclusion and Notes on the Future of the Project



## 1. Notes for the Reader/Viewer/Researcher
There are 2 versions of this project available. The github copy contains only the code and some notes on the changes made from 1 version of the code to the next. To access the full project, including the outputted data and models, see the VM provided by professor Skon or contact Adam S Blum. 

## 2. Overview of Project
The goal of this project was to solve a particular nonlinear poisson equation using physics informed neural networks (PINNs), and to document knowledge and wisdom about PINN training practices. This project was completed as coursework for the COMP401 class at Kenyon College during the spring 2025 semester. 
The equation to be solved was a 3D scalar field given by $$\nabla^2 [\phi(x,y,z)] + 2 \pi [\delta\rho(x,y,z)] [\phi(x,y,z)]^5 = 0$$, where $$[\delta\rho(x,y,z)]$$ is a provided 3d energy-related scalar field and $$[\phi(x,y,z)]$$ is the 'conformal factor' scalar field to be solved. Boundary conditions are periodic. This problem is called the Hamiltonian constraint for general relativity. My work is motivated by the computational cosmology research done by Kenyon Cosmolab under professor Tom Giblin. Solving these fields accurately and efficiently will help enable simulations of black hole formation in full general relativity. 
The project was successful, producing many good models for various delta-rho energy distributions (see attempt 7). The success levels of multiple training techniques were also recorded. 

## 3. Code/Training Versions Guide
This project contains 13 folders, labeled attempt 1 through attempt 13. Each folder contains code, with higher attempt numbers corresponding to newer training runs and newer versions of the code, and lower attempt numbers corresponding to older training runs and older versions of the code. Pre-training was achived by training with a pre-training script/code, then passing the produced model to the full training script/code. 
- Attempt 1: Initial code framework where I set up the neural network, physics functions, field sampling, learning rate schedulers, and loss formulation. Basic code for CPU, meant to be built upon in later versions.
- Attempt 2: Added data output features. Every 100 training iterations, the code outputs training to the screen. 
- Attempt 3: Added graphical visualization features and made it so that training data is saved to a csv every 100 training iterations. 
- Attempt 4: Added a second, alternate loss formulation using relative loss. PDE becomes $$\frac{\nabla^2 [\phi(x,y,z)] + 2 \pi [\delta\rho(x,y,z)] [\phi(x,y,z)]^5}{\nabla^2 [\phi(x,y,z)]} = 0$$. This was problematic and didn't function perfectly. Also implemented a delta-rho distribution (Gaussian/normal distribution about the origin). Completed a short test run of training. Didn't produce a decent model.
- Attempt 5: Added a pre-training feature, using the linear version of the full problem: $$\nabla^2 [\phi(x,y,z)] + 2 \pi [\delta\rho(x,y,z)] = 0$$. Did a test run of pre-training, then full training. Had some success! Model looks pretty good. 
- Attempt 6: Fixed relative loss formulation. Did more test runs with regular loss, NOT relative loss. 
- Attempt 7: Edited the code so that it would run correctly on the Vera-Rubin (HPC) GPU nodes. Did multiple test runs with pre-training. Implemented Gaussian distribution paramters that are relevent to the physics problem at hand. Found Success in 'train 2'. There were 30,000+ sampling points in the domain with an average MSE loss of order 10^(-8). We can also visually see that the solution has good radial symmetry, which is a feature that we expect from a good model.
- Attempt 8: Fully commented the code for readability.
- Attempt 9: Attempted a different Gaussian delta-rho distribution based on discussion with researchers from Kenyon Cosmolab. Didn't yield any good models. 
- Attempt 10: switched from floats to doubles for higher model loss precision. Test runs used the same delta-rho distribution from attempt 9 and failed to find any success. Was unable to choose an effective learning rate. Pre-training appears to have been bad in this case, causing a difficult-to-escape local minimum in the parameter space. 
- Attempt 11: Returned to the delta-rho distribution from attempt 7, where we found success. Attempted relative-loss-formulation post-training on the successful model from attempt 7. Was unable to reduce loss to a sufficiently low level. Relative loss formulation appears to have cause explosively large gradients that the MLA could not effectively work with. (More notes on this later). 
Note: an error was made and the relative loss formulation is for the linear problem here. The observed phenomena of explosive gradients is still relevent. 
- Attempt 12: Fixed relative loss forumlation from attempt 11. Still using the model from attempt 7. This attempt experimented with symmetry enforcemnt. Symmetry enforcemnt proved effective in guiding the model towards a decent shape. However, the model failed to reach sufficiently small loss, with or without symmetry enforcement. 
- Attempt 13 (WIP): Uses a delta-rho distribution similar to attempt 7, but with a larger magnitude, upon the requesyt of Kenyon Cosmolab members. Experiemented with a normal run and another run which used a combination of normal loss and relative loss. The normal run found moderate success with average MSE loss on the oder of 10^(-6) with 10,000+ sampling points. The run with relative loss failed to converge to a low-loss solution once again. 
- Exporters: This folder contains files used to export the delta-rho and phi fields for Kenyon Cosmolab research purposes.

## 4. Notes for the Casual Reader
For the most successful training run, see attempt 7, 'train 2'. 
For the most readable, up to date code, see attempt 13. 

## 5. Notes on Techniques Used
-  Pre-training using similar problems: Often helpful for achieving a general shape quickly, but usually unnecessary. Sometimes produced difficulties when the simplified pronblem pre-trained model was a weight-space local minimum for the full problem. Overall, helpful, but problematic in certain scenarios. 
- Symmetry enforcement: Useful for guiding chaotic nonlinear problems toward a valid solution during early training. On the downside, it increases computational cost and slows training. Overall, helpful for early parts of training where the model can't find the overall shape of the correct scalar field.
- Learning rate schedulers: Allows training to enter more suitable learning rate regimes when training has stalled out. Overall, extremely helpful, escpecially if you want to leave a training run going without checking on it and editing it frequently. Imporved my efficieny as a researcher. 
- Higher sampling near interesting dynamic areas (center of the domain, in this case): Helps models achieve an appropriate shape more quickly. This is probably the most important technique that I used. Potentially necissary for reducing computational cost to a feasible level, as opposed to completely random sampling on the domain. 
- Relative loss formulation: I have not yet found success with this technique. Dividing by the laplacian appears to make loss gradients too explosive. If learning rate is too large, the model becomes chaotic and unrecognizable, if the larning rate is too small, the model gets stuck in a local minimum with large loss. 
However, I haven't given up on this technique because both relative loss and regular loss should theoretically be 0 in a perfect model. So reducing relative loss would be a very good and useful measure of model success. I plan on adjusting my technique by setting a large lower limit on the divisor in this formulation, thus preventing divsion by small qunatities which would have otherwise lead to massive loss gradients. 

## 6. Conclusion and Notes on the Future of the Project
My time in the COMP401 class has come to an end, so I am submitting the project in it's current form. I have produced a good model that is currently being used for black hole research (attempt 7, 'train 2'). I have also recorded knowledge on multiple PINN techniques, that had varying levels of success. I have achieved what I sought out to achieve for the COMP401 project in computing. 
I intend to continue working on this project for my own interest and for the sake of Kenyon Cosmolab until I graduate in May or perhaps even after I graduate. My plan is to work with the delta-rho field from attempt 13, driving loss lower and hopefully implementing relative loss successfully. After I am done with the project, I will share my code and results with Kenyon Cosmolab, so that future student(s) may use it for further research. 
