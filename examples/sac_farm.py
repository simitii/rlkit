"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import gym
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp

from rlkit.envs.farmer import farmer as Farmer

import traceback

def acq_remote_env(farmer):
    # acquire a remote environment
    while True:
        remote_env = farmer.acq_env()
        if remote_env == False:  # no free environment
            pass
        else:
            break
    remote_env.set_spaces()
    print('action space', remote_env.action_space)
    print('observation space', remote_env.observation_space)
    return remote_env



def experiment(variant):

    farmlist_base = [('123.123.123.123', 4)]

    farmer = Farmer(farmlist_base)
    environment = acq_remote_env(farmer)
    env = NormalizedBoxEnv(environment)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        training_env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        environment_farming=True,
        farmlist_base=farmlist_base,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
