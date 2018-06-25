# pytorch_examples
This is an attempt at learning pytorch through their tutorial. Pre-requisites are knowing numpy and theoretical understanding of neural netwrks.
Below is the order in which files need to be looked into to make a step by step understanding:
  1. numpy_network.py - For building a 2 layer network and implementing backprop with numpy 
  2. tensor_network.py - For building a 2 layer network and implementing backprop with tensors which implies this code can run on a GPU
  3. autograd_network.py - For building a 2 layer network using tensors and letting the autograd package compute the gradients
  4. nn_networ.py - For building a 2 layer network and using nn package to define the network architecture while using autograd
  5. optim_nn_network.py - For building a 2 layer network and using optimizer to compute gradients
