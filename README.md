## Additions:
- Environment Farming

## Bugs:
- Dinamic allocation of Eval_Sampler(environment for it) - **FIXED**
- Environment allocation bugs - **FIXED** 
- Segmentation fault - **FIXED**

# rlkit
Reinforcement learning framework and algorithms implemented in PyTorch.

Some implemented algorithms:
 - Temporal Difference Models (TDMs)
    - [example script](examples/tdm/cheetah.py)
    - [TDM paper](https://arxiv.org/abs/1802.09081)
    - [Details on implementation](rlkit/torch/tdm/TDMs.md)
 - Deep Deterministic Policy Gradient (DDPG)
    - [example script](examples/ddpg.py)
    - [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
 - (Double) Deep Q-Network (DQN)
    - [example script](examples/dqn_and_double_dqn.py)
    - [DQN paper](https://arxiv.org/pdf/1509.06461.pdf)
    - [Double Q-learning paper](https://arxiv.org/pdf/1509.06461.pdf)
 - Soft Actor Critic (SAC)
    - [example script](examples/sac.py)
    - [SAC paper](https://arxiv.org/abs/1801.01290)
    - [TensorFlow implementation from author](https://github.com/haarnoja/sac)
 - Twin Dueling Deep Determinstic Policy Gradient (TD3)
    - [example script](examples/td3.py)
    - [TD3 paper](https://arxiv.org/abs/1802.09477)

To get started, checkout the example scripts, linked above.

## Installation
Install and use the included ananconda environment
```
$ conda env create -f rlkit-env.yml
$ source activate rlkit-env
```

## Credits
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
The serialization and logger code are basically a carbon copy of the rllab versions.
