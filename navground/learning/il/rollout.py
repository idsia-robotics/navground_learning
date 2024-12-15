# Patch of https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/data/rollout.py

from __future__ import annotations

import dataclasses
import inspect
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing
from imitation.data import rollout as rollout_without_info
from imitation.data import types
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import check_for_correct_spaces

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

Array: TypeAlias = np.typing.NDArray[Any]


class TrajectoryAccumulator(rollout_without_info.TrajectoryAccumulator):

    def add_steps_and_auto_finish(
        self,
        acts: Array,
        obs: types.Observation | dict[str, Array],
        rews: np.typing.NDArray[np.floating[Any]],
        dones: np.typing.NDArray[np.bool],
        infos: list[dict[str, Any]],
    ) -> list[types.TrajectoryWithRew]:
        trajs: list[types.TrajectoryWithRew] = []
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # iterate through environments
        for env_idx in range(len(wrapped_obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == [
                "obs"
            ], ("Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)")

        # iterate through steps
        zip_iter = enumerate(
            zip(acts, wrapped_obs, rews, dones, infos, strict=True))
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done and "terminal_observation" in info:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv, and
                # infos[i]["terminal_observation"] is the actual final observation.
                real_ob = types.maybe_wrap_in_dictobs(
                    info["terminal_observation"])
            else:
                real_ob = ob

            self.add_step(
                dict(
                    acts=act,
                    rews=rew,
                    # this is not the obs corresponding to `act`, but rather the obs
                    # *after* `act` (see above)
                    obs=real_ob,
                    infos=info,
                ),
                env_idx,
            )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                self.add_step(dict(obs=ob), env_idx)
        return trajs


# A PolicyCallableWithInfo is a function that takes an array of observations, an optional
# array of states, an optional array of episode starts, *and an optional list of infos*,
# and returns an array of corresponding actions.

PolicyCallableWithInfo: TypeAlias = Callable[
    [
        Array | dict[str, Array],  # observations
        tuple[Array, ...] | None,  # states
        np.typing.NDArray[np.bool] | None,  # episode_starts
        list[dict[str, Array]] | None  # (Jerome) added info
    ],
    tuple[Array, tuple[Array, ...] | None]  # actions, states
]

AnyPolicy: TypeAlias = rollout_without_info.AnyPolicy | PolicyCallableWithInfo


def policy_to_callable(
    policy: AnyPolicy,
    venv: VecEnv,
    deterministic_policy: bool = False,
) -> PolicyCallableWithInfo:
    """Converts any policy-like object into a function from observations to actions."""
    get_actions: PolicyCallableWithInfo
    if policy is None:

        def get_actions(
            observations: Array | dict[str, Array],
            states: tuple[Array, ...] | None,
            episode_starts: np.typing.NDArray[np.bool] | None,
            infos: list[dict[str, Array]] | None = None
        ) -> tuple[Array, tuple[Array, ...] | None]:
            acts = [
                venv.action_space.sample() for _ in range(len(observations))
            ]
            return np.stack(acts, axis=0), None

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):
        # There's an important subtlety here: BaseAlgorithm and BasePolicy
        # are themselves Callable (which we check next). But in their case,
        # we want to use the .predict() method, rather than __call__()
        # (which would call .forward()). So this elif clause must come first!

        def get_actions(
            observations: Array | dict[str, Array],
            states: tuple[Array, ...] | None,
            episode_starts: np.typing.NDArray[np.bool] | None,
            infos: list[dict[str, Array]] | None = None
        ) -> tuple[Array, tuple[Array, ...] | None]:
            assert isinstance(policy, (BaseAlgorithm, BasePolicy))
            # pytype doesn't seem to understand that policy is a BaseAlgorithm
            # or BasePolicy here, rather than a Callable
            (acts, states) = policy.predict(  # pytype: disable=attribute-error
                observations,
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic_policy,
            )
            return acts, states

    elif callable(policy):
        # When a policy callable is passed, by default we will use it directly.
        # We are not able to change the determinism of the policy when it is a
        # callable that only takes in the states.
        if deterministic_policy:
            # raise ValueError(
            warnings.warn(
                "Cannot set deterministic_policy=True when policy is a callable, "
                "since deterministic_policy argument is ignored.",
                stacklevel=1)

        sig = inspect.signature(policy)
        if len(sig.parameters) == 3:

            def get_actions(
                observations: Array | dict[str, Array],
                states: tuple[Array, ...] | None,
                episode_starts: np.typing.NDArray[np.bool] | None,
                infos: list[dict[str, Array]] | None = None
            ) -> tuple[Array, tuple[Array, ...] | None]:
                return cast(rollout_without_info.PolicyCallable,
                            policy)(observations, states, episode_starts)
        else:
            get_actions = cast(PolicyCallableWithInfo, policy)
    else:
        raise TypeError(
            "Policy must be None, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead", )

    if isinstance(policy, BaseAlgorithm):
        # Check that the observation and action spaces of policy and environment match
        try:
            check_for_correct_spaces(
                venv,
                policy.observation_space,
                policy.action_space,
            )
        except ValueError as e:
            # Check for a particularly common mistake when using image environments.
            venv_obs_shape = venv.observation_space.shape
            assert policy.observation_space is not None
            policy_obs_shape = policy.observation_space.shape
            assert venv_obs_shape is not None
            assert policy_obs_shape is not None
            if len(venv_obs_shape) != 3 or len(policy_obs_shape) != 3:
                raise e
            venv_obs_rearranged = (
                venv_obs_shape[2],
                venv_obs_shape[0],
                venv_obs_shape[1],
            )
            if venv_obs_rearranged != policy_obs_shape:
                raise e
            raise ValueError(
                "Policy and environment observation shape mismatch. "
                "This is likely caused by "
                "https://github.com/HumanCompatibleAI/imitation/issues/599. "
                "If encountering this from rollout.rollout, try calling:\n"
                "rollout.rollout(expert, expert.get_env(), ...) instead of\n"
                "rollout.rollout(expert, env, ...)\n\n"
                f"Policy observation shape: {policy_obs_shape} \n"
                f"Environment observation shape: {venv_obs_shape}") from e

    return get_actions


def generate_trajectories(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: rollout_without_info.GenTrajTerminationFn,
    rng: np.random.Generator,
    *,
    deterministic_policy: bool = False,
) -> Sequence[types.TrajectoryWithRew]:
    """
    Patches the orginal version in ``imitation``
    to add support for :py:type:`PolicyCallableWithInfo`

    :param      policy:                The policy
    :param      venv:                  The venv
    :param      sample_until:          The sample until criteria
    :param      rng:                   The random number generator
    :param      deterministic_policy:  Whether the policy is deterministic

    :returns:   the trajectories
    """
    get_actions = policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator(
    )  # type: ignore[no-untyped-call]
    obs = venv.reset()
    infos = venv.reset_infos
    assert isinstance(
        obs, (np.ndarray, dict)), "Tuple observations are not supported."
    wrapped_obs = types.maybe_wrap_in_dictobs(obs)

    # we use dictobs to iterate over the envs in a vecenv
    for env_idx, ob in enumerate(wrapped_obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    state = None
    dones = np.zeros(venv.num_envs, dtype=bool)
    while np.any(active):
        if any(dones):
            infos = list(infos)
            for i, v in enumerate(dones):
                if v:
                    infos[i] = venv.reset_infos[i]
        # policy gets unwrapped observations (eg as dict, not dictobs)
        acts, state = get_actions(obs, state, dones, infos)
        obs, rews, dones, infos = venv.step(acts)
        dones = dones.astype(bool)
        assert isinstance(
            obs,
            (np.ndarray, dict),
        ), "Tuple observations are not supported."
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            wrapped_obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        if isinstance(venv.observation_space, gym.spaces.Dict):
            exp_obs = {}
            for k, v in venv.observation_space.items():
                assert v.shape is not None
                exp_obs[k] = (n_steps + 1, ) + v.shape
        else:
            obs_space_shape = venv.observation_space.shape
            assert obs_space_shape is not None
            exp_obs = (n_steps +
                       1, ) + obs_space_shape  # type: ignore[assignment]
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        assert venv.action_space.shape is not None
        exp_act = (n_steps, ) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps, )
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories


def generate_transitions(
    policy: AnyPolicy,
    venv: VecEnv,
    n_timesteps: int,
    rng: np.random.Generator,
    *,
    truncate: bool = True,
    **kwargs: Any,
) -> types.TransitionsWithRew:
    traj = generate_trajectories(
        policy,
        venv,
        sample_until=rollout_without_info.make_min_timesteps(n_timesteps),
        rng=rng,
        **kwargs,
    )
    transitions = rollout_without_info.flatten_trajectories_with_rew(traj)
    if truncate and n_timesteps is not None:
        as_dict = types.dataclass_quick_asdict(transitions)
        truncated = {k: arr[:n_timesteps] for k, arr in as_dict.items()}
        transitions = types.TransitionsWithRew(**truncated)
    return transitions


def rollout(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: rollout_without_info.GenTrajTerminationFn,
    rng: np.random.Generator,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Sequence[types.TrajectoryWithRew]:
    trajs = generate_trajectories(
        policy,
        venv,
        sample_until,
        rng=rng,
        **kwargs,
    )
    if unwrap:
        trajs = [rollout_without_info.unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout_without_info.rollout_stats(trajs)
        logging.info(f"Rollout stats: {stats}")
    return trajs
