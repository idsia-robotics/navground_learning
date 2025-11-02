from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import gymnasium as gym
    import torch
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

from navground import core

from ...config import ControlActionConfig, DefaultObservationConfig
from ...internal.group import GymAgent
from ...types import (Action, AnyPolicyPredictor, Observation,
                      ObservationTransform, PathLike)


class BasePolicyMixin:
    """
    A navigation behavior that evaluates a ML policy

    *Registered properties*:

    - :py:attr:`policy_path` (str)
    - :py:attr:`flat` (bool)
    - :py:attr:`history` (int)
    - :py:attr:`fix_orientation` (bool)
    - :py:attr:`include_target_direction` (bool)
    - :py:attr:`include_target_direction_validity` (bool)
    - :py:attr:`include_target_distance` (bool)
    - :py:attr:`include_target_distance_validity` (bool)
    - :py:attr:`include_target_speed` (bool)
    - :py:attr:`include_target_angular_speed` (bool)
    - :py:attr:`include_velocity` (bool)
    - :py:attr:`include_angular_speed` (bool)
    - :py:attr:`include_radius` (bool)
    - :py:attr:`use_wheels` (bool)
    - :py:attr:`use_acceleration_action` (bool)
    - :py:attr:`max_acceleration` (float)
    - :py:attr:`max_angular_acceleration` (float)
    - :py:attr:`deterministic` (bool)

    *State*: :py:class:`core.SensingState`

    :param kinematics: The agent kinematics
    :param radius: The agent radius
    :param policy: The policy
    :param policy_path: The policy path
    :param action_config: Configures which actions the policy generates
    :param observation_config: Configures which observations the policy consumes
    :param deterministic: Whether the policy evaluation is deterministic
    :param pre: Optional input (observations) transformation
    """

    _policies: dict[pathlib.Path, AnyPolicyPredictor] = {}
    _cache_is_enabled: bool = False

    def __init__(self,
                 policy: AnyPolicyPredictor | None = None,
                 policy_path: PathLike = '',
                 action_config: ControlActionConfig = ControlActionConfig(),
                 observation_config:
                 DefaultObservationConfig = DefaultObservationConfig(),
                 deterministic: bool = False,
                 pre: ObservationTransform | None = None):
        self._state = core.SensingState()
        self._policy = policy
        self._policy_path: pathlib.Path | None = None
        self._gym_agent: GymAgent | None = None
        self._action_config = action_config
        self._observation_config = observation_config
        self._deterministic = deterministic
        self._pre = pre

    @classmethod
    def enable_cache(cls, value: bool) -> None:
        cls._cache_is_enabled = value

    @classmethod
    def cache_is_enabled(cls) -> bool:
        return cls._cache_is_enabled

    @classmethod
    def reset_cache(cls) -> None:
        cls._policies.clear()

    def init_policy(self) -> None:
        """
        Loads the policy from :py:attr:`policy_path` if not already done.
        """
        if not self._policy and self._policy_path:
            self._policy = self.load_policy(self._policy_path)

    @classmethod
    def load_policy(cls, path: str | pathlib.Path) -> AnyPolicyPredictor:
        if isinstance(path, str):
            path = pathlib.Path(path)
        if cls.cache_is_enabled():
            path = path.absolute()
            if path not in cls._policies:
                cls._policies[path] = cls._load_policy(path)
            return cls._policies[path]
        return cls._load_policy(path)

    @classmethod
    def _load_policy(cls, path: pathlib.Path) -> AnyPolicyPredictor:
        if path.suffix == '.onnx':
            return cls._load_onnx_policy(path)
        return cls._load_torch_policy(path)

    @classmethod
    def _load_onnx_policy(cls, path: pathlib.Path) -> AnyPolicyPredictor:
        from ...onnx.policy import OnnxPolicy

        return OnnxPolicy(path)

    @classmethod
    def _load_torch_policy(cls, path: pathlib.Path) -> AnyPolicyPredictor:
        import torch
        from stable_baselines3.common.policies import BasePolicy
        try:
            model = torch.load(path, weights_only=False)
            if isinstance(model, BasePolicy):
                return model
            raise RuntimeError("Loaded model is not a policy")
        except Exception as e:
            raise RuntimeError(f"Could not load policy from {path}") from e

    @property
    @core.register("", "policy path")
    def policy_path(self) -> str:
        """
        The file from which to load the model
        """
        if not self._policy_path:
            return ''
        if sys.version_info >= (3, 12):
            return str(
                self._policy_path.relative_to(pathlib.Path.cwd(), walk_up=True))
        else:
            return str(
                self._policy_path.relative_to(pathlib.Path.cwd()))
        # return str(self._policy_path)

    @policy_path.setter
    def policy_path(self, value: PathLike) -> None:
        self.set_policy_path(value, load_policy=False)

    def set_policy_path(self,
                        value: PathLike,
                        load_policy: bool = False) -> None:
        value = pathlib.Path(value).absolute()
        try:
            if self._policy_path and self._policy_path.samefile(value):
                return
        except FileNotFoundError:
            return
        self._policy_path = value
        self._policy = None
        if load_policy:
            self.init_policy()

    @property
    @core.register(
        False, "Whether to include the target distance in the observations")
    def include_target_distance(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_target_distance`
        """
        return self._observation_config.include_target_distance

    @include_target_distance.setter
    def include_target_distance(self, value: bool) -> None:
        self._observation_config.include_target_distance = value

    @property
    @core.register(
        False,
        "Whether to include the target distance validity in the observations")
    def include_target_distance_validity(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_target_distance_validity`
        """
        return self._observation_config.include_target_distance_validity

    @include_target_distance_validity.setter
    def include_target_distance_validity(self, value: bool) -> None:
        self._observation_config.include_target_distance_validity = value

    @property
    @core.register(
        True, "Whether to include the target direction in the observations")
    def include_target_direction(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_target_direction`
        """
        return self._observation_config.include_target_direction

    @include_target_direction.setter
    def include_target_direction(self, value: bool) -> None:
        self._observation_config.include_target_direction = value

    @property
    @core.register(
        False,
        "Whether to include the target direction validity in the observations")
    def include_target_direction_validity(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_target_direction_validity`
        """
        return self._observation_config.include_target_direction_validity

    @include_target_direction_validity.setter
    def include_target_direction_validity(self, value: bool) -> None:
        self._observation_config.include_target_direction_validity = value

    @property
    @core.register(False,
                   "Whether to include the target speed in the observations")
    def include_target_speed(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_target_speed`
        """
        return self._observation_config.include_target_speed

    @include_target_speed.setter
    def include_target_speed(self, value: bool) -> None:
        self._observation_config.include_target_speed = value

    @property
    @core.register(False,
                   "Whether to include the target speed in the observations")
    def include_target_angular_speed(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_target_angular_speed`
        """
        return self._observation_config.include_target_angular_speed

    @include_target_angular_speed.setter
    def include_target_angular_speed(self, value: bool) -> None:
        self._observation_config.include_target_angular_speed = value

    @property
    @core.register(
        False, "Whether to include the current velocity in the observations")
    def include_velocity(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_velocity`
        """
        return self._observation_config.include_velocity

    @include_velocity.setter
    def include_velocity(self, value: bool) -> None:
        self._observation_config.include_velocity = value

    @property
    @core.register(
        False,
        "Whether to include the current angular speed in the observations")
    def include_angular_speed(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_angular_speed`
        """
        return self._observation_config.include_angular_speed

    @include_angular_speed.setter
    def include_angular_speed(self, value: bool) -> None:
        self._observation_config.include_angular_speed = value

    @property
    @core.register(False,
                   "Whether to include the own radius in the observations")
    def include_radius(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.include_radius`
        """
        return self._observation_config.include_radius

    @include_radius.setter
    def include_radius(self, value: bool) -> None:
        self._observation_config.include_radius = value

    @property
    @core.register(False, "Whether to flatten the observations")
    def flat(self) -> bool:
        """
        See :py:attr:`.DefaultObservationConfig.flat`
        """
        return self._observation_config.flat

    @flat.setter
    def flat(self, value: bool) -> None:
        self._observation_config.flat = value

    @property
    @core.register(1, "Length of the queue")
    def history(self) -> int:
        """
        See :py:attr:`.DefaultObservationConfig.history`
        """
        return self._observation_config.history

    @history.setter
    def history(self, value: int) -> None:
        self._observation_config.history = value

    @property
    @core.register(False, "Whether to keep orientation fixed")
    def fix_orientation(self) -> bool:
        """
        See :py:attr:`.ControlActionConfig.fix_orientation`
        """
        return self._action_config.fix_orientation

    @fix_orientation.setter
    def fix_orientation(self, value: bool) -> None:
        self._action_config.fix_orientation = value

    @property
    @core.register(
        False,
        "Whether action and observation uses wheel speeds or accelerations")
    def use_wheels(self) -> bool:
        """
        See :py:attr:`.ControlActionConfig.use_wheels`
        """
        return self._action_config.use_wheels

    @use_wheels.setter
    def use_wheels(self, value: bool) -> None:
        self._action_config.use_wheels = value

    @property
    @core.register(False, "Whether actions are accelerations.")
    def use_acceleration_action(self) -> bool:
        """
        See :py:attr:`.ControlActionConfig.use_acceleration_action`
        """
        return self._action_config.use_acceleration_action

    @use_acceleration_action.setter
    def use_acceleration_action(self, value: bool) -> None:
        self._action_config.use_acceleration_action = value

    @property
    @core.register(10.0, "The upper bound of the acceleration.")
    def max_acceleration(self) -> float:
        """
        See :py:attr:`.ControlActionConfig.max_acceleration`
        """
        return self._action_config.max_acceleration

    @max_acceleration.setter
    def max_acceleration(self, value: float) -> None:
        self._action_config.max_acceleration = value

    @property
    @core.register(100.0, "The upper bound of the angular acceleration.")
    def max_angular_acceleration(self) -> float:
        """
        See :py:attr:`.ControlActionConfig.max_angular_acceleration`
        """
        return self._action_config.max_angular_acceleration

    @max_angular_acceleration.setter
    def max_angular_acceleration(self, value: float) -> None:
        self._action_config.max_angular_acceleration = value

    @property
    @core.register(False, "Whether or not to output deterministic actions")
    def deterministic(self) -> bool:
        """
        Whether or not to output deterministic actions
        """
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        self._deterministic = value

    def get_environment_state(self) -> core.SensingState:
        return self._state

    def prepare(self) -> None:
        if not self._gym_agent:
            self._gym_agent = GymAgent(observation=self._observation_config,
                                       action=self._action_config,
                                       behavior=cast('core.Behavior', self))

    def update_observation(self) -> Observation:
        if not self._gym_agent:
            raise RuntimeError("Not configured")
        return self._gym_agent.update_observation()

    def get_cmd_from_action(self, action: Action,
                            time_step: float) -> core.Twist2:
        if not self._gym_agent:
            raise RuntimeError("Not configured")
        return cast('core.Behavior', self).feasible_twist(
            self._gym_agent.get_cmd_from_action(action, time_step))

    @classmethod
    def clone_behavior(cls,
                       behavior: core.Behavior,
                       policy: AnyPolicyPredictor | PathLike | None,
                       action_config: ControlActionConfig,
                       observation_config: DefaultObservationConfig,
                       deterministic: bool = False,
                       pre: ObservationTransform | None = None) -> Self:
        """
        Configure a new policy behavior from a behavior.

        :param      behavior:         The behavior to replicate
        :param      policy:           The policy
        :param      config:           The configuration
        :param      deterministic:    Whether or not to output deterministic actions
        :param pre: Optional input (observations) transformation

        :returns:   The configured policy behavior
        """
        try:
            policy_path: PathLike = pathlib.Path(
                policy)  # type: ignore[arg-type]
            policy = None
        except TypeError:
            policy_path = ''
        pb = cls(policy=cast("AnyPolicyPredictor | None", policy),
                 action_config=action_config,
                 observation_config=observation_config,
                 deterministic=deterministic,
                 pre=pre)
        if policy_path:
            pb.set_policy_path(policy_path, load_policy=True)
        cast('core.Behavior', pb).set_state_from(behavior)
        return pb

    def clone(self) -> Self:
        """
        Creates a new policy behavior with same properties but a separate state.

        :returns:   Copy of this object.
        """
        return type(self).clone_behavior(
            behavior=cast('core.Behavior', self),
            policy=self._policy,
            action_config=self._action_config,
            observation_config=self._observation_config,
            deterministic=self._deterministic)
