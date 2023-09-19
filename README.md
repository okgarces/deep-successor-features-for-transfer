# Transformed Successor Features

This code is a fork of Gimelfarb repository. Has the following changes:
1. Use PyTorch instead of Tensorflow.
2. Implement Transformed Successor Features.
3. Refactor to support only one file.

Currently supports:
- deep neural network SF representations for large or continuous-state environments.
- tasks with pre-defined state features only, although support for training features on-the-fly may be added later
- tasks structured according to the OpenAI gym framework

Current tasks:
- Reacher environment with 7 discrete actions
- Hopper with 27 discrete actions
- Cartpole v2.0

# Requirements
- python 3.8 or later
- tensorflow 2.3 or later
- pybullet 3.0.8 and pybullet-gym 0.1 (for reacher domain)

# References
[1] Barreto, André, et al. "Successor features for transfer in reinforcement learning." Advances in neural information processing systems. 2017.
[2] Dayan, Peter. "Improving generalization for temporal difference learning: The successor representation." Neural Computation 5.4 (1993): 613-624.
