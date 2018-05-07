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

class Deneme:
    def step(self,*argv,**kargs):
        print("step is called!")
        #traceback.print_stack()
        observation = np.random.rand(30)
        reward = np.random.rand(1)
        done =False
        info = "poku yedik"
        return observation, reward,done,info
    def reset(self,*argv,**kargs):
        print("reset is called!")
        #traceback.print_stack()
        observation = np.random.rand(30)
        return observation

    def __init__(self):
        noutput = 18
        action_space = [0.0] * noutput, [1.0] * noutput
        action_space = gym.spaces.Box(np.array(action_space[0]), np.array(action_space[1]))
        self.action_space = action_space
        noutput = 30
        observation_space = [0.0] * noutput, [1.0] * noutput
        observation_space = gym.spaces.Box(np.array(observation_space[0]), np.array(observation_space[1]))
        self.observation_space = observation_space

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

    #TODO environment.release()

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
