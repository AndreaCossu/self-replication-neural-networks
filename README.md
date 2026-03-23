# Self Replication in Neural Networks

Unofficial code for some of the experiments of the paper "Self-Replication in Neural Networks" by Gabor et al. (Artificial Life, 2022) https://doi.org/10.1162/artl_a_00359

Short story short: I was curious about the paper, did not find the code, decided to write one in Jax.  
Not super-polished, yet.

## How to use

Pre-requisites: jax, optax, matplotlib, numpy, scikit-learn

Two main scripts, depending on what you want to do:
* quine.py --> use this for weightwise (self)-application N <- M or N <- N
* train_quine.py -> use this for weightwise (self)-training N <-w M or N <-w N (can alternate between a few steps of training and a few steps of application).

Each script produces a PCA plot of the parameters of the N network over time.

To run: python \[your script.py\] [args] 

The command-line arguments are only a few (`python script.py --help` to list them).  
Inspecting the script requires a few minutes.




