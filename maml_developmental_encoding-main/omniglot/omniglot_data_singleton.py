

import numpy as np
import os.path
import random

# This module is used like a singleton
# Only import it if you want to load the dataset into memory.
# On repeated import, the dataset is not loaded again.
# This is so dask workers not load the whole dataset on every invocation. 
# (we use a functional style invocation, modules are a way to store worker state)

# It is around a 100 Mb 
#if os.path.exists("/users/ak1774/scratch/ES_MAML/omniglot_data/omniglot.npy"):
#    dataset = np.load("/users/ak1774/scratch/ES_MAML/omniglot_data/omniglot.npy")
#else:
#    dataset = np.load("/home/userfs/a/ak1774/workspace/omniglot_dataset/omniglot.npy")
    
if os.path.exists("/content/gdrive/My Drive/maml_developmental_encoding-main/images/omniglot.npy"):
    dataset = np.load("/content/gdrive/My Drive/maml_developmental_encoding-main/images/omniglot.npy")
else:
    dataset = np.load("../../images/omniglot.npy")

dataset = 1-dataset # invert the images, same as orig maml

omniglot_shuffled_indicies = list(range(dataset.shape[0]))

random.seed(1)
random.shuffle(omniglot_shuffled_indicies)
random.seed() # reset random with a random seed