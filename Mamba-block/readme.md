# Mamba Block: Integration Guide
This guide offers instructions for integrating the Mamba and FusionMamba blocks into the network.

## Installation
1. Install the official Mamba library by following the instructions in the [hustvl/Vim](https://github.com/hustvl/Vim) repository. The head should be point at `b9cf48f`, to achieve this, by run `git checkout b9cf48f`, then install the Vim by following the repo at b9cf48f. In our practice, it's necessary to add `pip install causal_conv1d==1.1.0` after completing all installation steps in Vim.
2. After installing the official Mamba library, replace the mamba_simpy.py file in the installation directory (./Vim/mamba-1p1p1/mamba_ssm/modules/) with the one provided in this mamba block directory.
3. Please do not install Mamba directly using `pip install mamba-ssm`!

## Explanation
We provide a novel method for combining two types of data, both of the same size, leveraging the theory of state space models (SSM). This approach is applicable to various fusion tasks and can be seamlessly integrated into your existing projects.
