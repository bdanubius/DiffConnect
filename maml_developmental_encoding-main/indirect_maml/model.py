
import torch
import numpy as np


# In this project we are interested in indirectly encoded neural networks, meaning that the learned parameters are transformed 
# Also we want to handle models where parts of the model in indirectly encoded, and parts of it is directly encoded
# All the parameters are collected in 3 parameter dicts
# - generator_parameters         (parameters which are used to generate the parameters of the network)
# - directly_encoded_parameters  (parameters which are used in the network without any modification or transformation)
# - generated_weights            (parameters which were generated using the generator_parameters)

# Also there is:
# - net_parameters               (the merged dict of directly_encoded_parameters and generated_weights, the parameters needed to use the network)

# With different algorithms, we might update different parameters


# To define a model you need to provide 3 functions
# get_model_fn() -> generator_parameters,directly_encoded_parameters
# generate_fn(generator_parameters) -> generated_weights
# forward_fn(x,net_parameters) -> network_output



##############################
# Directly encoded MNIST net #
##############################
# For now it is hardcoded to 784 x 784 x 10



def get_MNIST_direct_encoding_parameters(device=None):
    
    if device is None:
        device = torch.device("cpu")
    
    generator_parameters = {}
    directly_encoded_parameters = {
        "W0" : torch.nn.Parameter(torch.randn(784,784).to(device)*np.sqrt(2/784)),
        "W1" : torch.nn.Parameter(torch.randn(784,10).to(device)*np.sqrt(2/784)),
        "B0" : torch.nn.Parameter(torch.zeros(784).to(device)),
        "B1" : torch.nn.Parameter(torch.zeros(10).to(device)),
    }
        
    return generator_parameters,directly_encoded_parameters

def dummy_generate_net(generator_parameters):
    return {}



def get_MNIST_simple_GEM_parameters(device=None):
    """
    This function returns the parameters of the indirectly encoded model
    There are 2 kinds of parameters:
     - generator_parameters   they are the parameters which are used to generate (develop) the weights of the direct network
                              These parameters should not be finetuned in when doing indirect maml with direct finetuning
     - directly_encoded_parameters   These are parameters are part of the direct network which we dont wish to generate,
                                     but to encode them directly (for example biases). These parameters should be finetuned.
    """
    if device is None:
        device = torch.device("cpu")
        
    generator_parameters = {
        "X0" : torch.nn.Parameter(torch.randn(784,10).to(device)* 0.1),# i dont know what is the correct scaling, 0.1 seems all right
        "O0" : torch.nn.Parameter(torch.randn(10,10).to(device)* 0.1), # by correct scaling i mean so the generated net have 1 gain
        "X1" : torch.nn.Parameter(torch.randn(784,10).to(device)* 0.1),
        "O1" : torch.nn.Parameter(torch.randn(10,10).to(device)* 0.1),
        "X2" : torch.nn.Parameter(torch.randn(10,10).to(device)* 0.1),
    }
    
    directly_encoded_parameters = {
        "B0" : torch.nn.Parameter(torch.zeros(784).to(device)),
        "B1" : torch.nn.Parameter(torch.zeros(10).to(device)),
    }
    
    return generator_parameters,directly_encoded_parameters

def generate_MNIST_net_simple_GEM(generator_parameters):
    G = generator_parameters # alias for shorter code
    
    generated_weights = {
        "W0" : torch.matmul(torch.matmul(G["X0"],G["O0"]),G["X1"].T),
        "W1" : torch.matmul(torch.matmul(G["X1"],G["O1"]),G["X2"].T),
    }
    
    return generated_weights


def forward_MNIST_net(x,net_parameters):
    x = torch.nn.functional.linear(x, net_parameters["W0"].T, net_parameters["B0"])
    x = torch.relu(x)
    x = torch.nn.functional.linear(x, net_parameters["W1"].T, net_parameters["B1"])
    return x
    


