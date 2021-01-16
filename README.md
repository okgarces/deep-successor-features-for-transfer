# successor-features-for-transfer
A reusable framework for successor features (SF) for transfer in (deep) reinforcement learning using keras, and based on [1].

Currently supports:
- tabular SF representations for discrete environments, based on an efficient hash table representation
- deep neural network SF representations for large or continuous-state environments, based on keras; allows existing keras models or custom architectures (e.g. CNNs) as inputs for easy training and tuning
- tasks with pre-defined state features only, although support for training features on-the-fly may be added later

# Prerequisites
This project has been tested on Python 3.8.5 with the latest version of tensorflow (at least 2.3). 

# References
[1] Barreto, André, et al. "Successor features for transfer in reinforcement learning." Advances in neural information processing systems. 2017.
[2] Dayan, Peter. "Improving generalization for temporal difference learning: The successor representation." Neural Computation 5.4 (1993): 613-624.