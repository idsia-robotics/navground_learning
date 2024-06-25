import pettingzoo as pz
import supersuit
from stable_baselines3.common.vec_env import VecEnv


def make_venv(env: pz.ParallelEnv,
              number: int = 1,
              processes: int = 1) -> VecEnv:
    penv = supersuit.pettingzoo_env_to_vec_env_v1(env)
    penv = supersuit.concat_vec_envs_v1(penv,
                                        number,
                                        num_cpus=processes,
                                        base_class="stable_baselines3")
    return penv
