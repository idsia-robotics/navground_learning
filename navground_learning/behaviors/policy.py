import logging
import pathlib
from typing import Any

from navground import core

from ..core import ControlActionConfig, GymAgent, ObservationConfig


class PolicyBehavior(core.Behavior, name="Policy"):
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
    - :py:attr:`include_radius` (bool)
    - :py:attr:`use_wheels` (bool)
    - :py:attr:`use_acceleration_action` (bool)
    - :py:attr:`max_acceleration` (float)
    - :py:attr:`max_angular_acceleration` (float)
    - :py:attr:`deterministic` (bool)

    *State*: :py:class:`SensingState`

    :param kinematics: The agent kinematics
    :param radius: The agent radius
    :param policy: The policy
    :param config: How to use the policy (default if not specified)
    """

    _policies: dict[str, Any] = {}

    def __init__(self,
                 kinematics: core.Kinematics | None = None,
                 radius: float = 0.0,
                 policy: Any = None,
                 action_config: ControlActionConfig = ControlActionConfig(),
                 observation_config: ObservationConfig = ObservationConfig(),
                 deterministic: bool = False):
        super().__init__(kinematics, radius)
        self._state = core.SensingState()
        self._policy = policy
        self._policy_path = ""
        self._gym_agent: GymAgent | None = None
        self._action_config = action_config
        self._observation_config = observation_config
        self._deterministic = deterministic

    @classmethod
    def load_policy(cls, path: str) -> Any:
        import torch

        if path not in cls._policies:
            try:
                cls._policies[path] = torch.load(path)
            except Exception as e:
                logging.error(f"Could not load policy from {path}: {e}")
        return cls._policies.get(path)

    @property
    @core.register("", "policy path")
    def policy_path(self) -> str:
        """
        The file from which to load the model
        """
        return self._policy_path

    @policy_path.setter  # type: ignore[no-redef]
    def policy_path(self, value: str) -> None:
        if not value or self._policy_path == value:
            return
        self._policy = PolicyBehavior.load_policy(value)
        self._policy_path = value

    def reset_policy_path(self, value: str | pathlib.Path) -> None:
        self._policy_path = str(value)

    @property
    @core.register(
        True, "Whether to include the target distance in the observations")
    def include_target_distance(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_distance`
        """
        return self._observation_config.include_target_distance

    @include_target_distance.setter  # type: ignore[no-redef]
    def include_target_distance(self, value: bool) -> None:
        self._observation_config.include_target_distance = value

    @property
    @core.register(
        True,
        "Whether to include the target distance validity in the observations")
    def include_target_distance_validity(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_distance_validity`
        """
        return self._observation_config.include_target_distance_validity

    @include_target_distance_validity.setter  # type: ignore[no-redef]
    def include_target_distance_validity(self, value: bool) -> None:
        self._observation_config.include_target_distance_validity = value

    @property
    @core.register(
        True, "Whether to include the target direction in the observations")
    def include_target_direction(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_direction`
        """
        return self._observation_config.include_target_direction

    @include_target_direction.setter  # type: ignore[no-redef]
    def include_target_direction(self, value: bool) -> None:
        self._observation_config.include_target_direction = value

    @property
    @core.register(
        True,
        "Whether to include the target direction validity in the observations")
    def include_target_direction_validity(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_direction_validity`
        """
        return self._observation_config.include_target_direction_validity

    @include_target_direction_validity.setter  # type: ignore[no-redef]
    def include_target_direction_validity(self, value: bool) -> None:
        self._observation_config.include_target_direction_validity = value

    @property
    @core.register(False,
                   "Whether to include the target speed in the observations")
    def include_target_speed(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_speed`
        """
        return self._observation_config.include_target_speed

    @include_target_speed.setter  # type: ignore[no-redef]
    def include_target_speed(self, value: bool) -> None:
        self._observation_config.include_target_speed = value

    @property
    @core.register(False,
                   "Whether to include the target speed in the observations")
    def include_target_angular_speed(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_angular_speed`
        """
        return self._observation_config.include_target_angular_speed

    @include_target_angular_speed.setter  # type: ignore[no-redef]
    def include_target_angular_speed(self, value: bool) -> None:
        self._observation_config.include_target_angular_speed = value

    @property
    @core.register(
        False, "Whether to include the current velocity in the observations")
    def include_velocity(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_velocity`
        """
        return self._observation_config.include_velocity

    @include_velocity.setter  # type: ignore[no-redef]
    def include_velocity(self, value: bool) -> None:
        self._observation_config.include_velocity = value

    @property
    @core.register(False,
                   "Whether to include the own radius in the observations")
    def include_radius(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_radius`
        """
        return self._observation_config.include_radius

    @include_radius.setter  # type: ignore[no-redef]
    def include_radius(self, value: bool) -> None:
        self._observation_config.include_radius = value

    @property
    @core.register(False, "Whether to flatten the observations")
    def flat(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.flat`
        """
        return self._observation_config.flat

    @flat.setter  # type: ignore[no-redef]
    def flat(self, value: bool) -> None:
        self._observation_config.flat = value

    @property
    @core.register(False, "Length of the queue")
    def history(self) -> int:
        """
        See :py:attr:`GymAgentConfig.history`
        """
        return self._observation_config.history

    @history.setter  # type: ignore[no-redef]
    def history(self, value: int) -> None:
        self._observation_config.history = value

    @property
    @core.register(False, "Whether to keep orientation fixed")
    def fix_orientation(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.fix_orientation`
        """
        return self._action_config.fix_orientation

    @fix_orientation.setter  # type: ignore[no-redef]
    def fix_orientation(self, value: bool) -> None:
        self._action_config.fix_orientation = value

    @property
    @core.register(
        False,
        "Whether action and observation uses wheel speeds or accelerations")
    def use_wheels(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.use_wheels`
        """
        return self._action_config.use_wheels

    @use_wheels.setter  # type: ignore[no-redef]
    def use_wheels(self, value: bool) -> None:
        self._action_config.use_wheels = value

    @property
    @core.register(False, "Whether actions are accelerations.")
    def use_acceleration_action(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.use_acceleration_action`
        """
        return self._action_config.use_acceleration_action

    @use_acceleration_action.setter  # type: ignore[no-redef]
    def use_acceleration_action(self, value: bool) -> None:
        self._action_config.use_acceleration_action = value

    @property
    @core.register(10.0, "The upper bound of the acceleration.")
    def max_acceleration(self) -> float:
        """
        See :py:attr:`GymAgentConfig.max_acceleration`
        """
        return self._action_config.max_acceleration

    @max_acceleration.setter  # type: ignore[no-redef]
    def max_acceleration(self, value: float) -> None:
        self._action_config.max_acceleration = value

    @property
    @core.register(100.0, "The upper bound of the angular acceleration.")
    def max_angular_acceleration(self) -> float:
        """
        See :py:attr:`GymAgentConfig.max_angular_acceleration`
        """
        return self._action_config.max_angular_acceleration

    @max_angular_acceleration.setter  # type: ignore[no-redef]
    def max_angular_acceleration(self, value: float) -> None:
        self._action_config.max_angular_acceleration = value

    @property
    @core.register(False, "Whether or not to output deterministic actions")
    def deterministic(self) -> bool:
        """
        Whether or not to output deterministic actions
        """
        return self._deterministic

    @deterministic.setter  # type: ignore[no-redef]
    def deterministic(self, value: bool) -> None:
        self._deterministic = value

    def get_environment_state(self) -> core.SensingState:
        return self._state

    def compute_cmd_internal(self, time_step: float) -> core.Twist2:
        if self._policy is None:
            return core.Twist2((0, 0), 0, frame=core.Frame.relative)
        if not self._gym_agent:
            self._gym_agent = GymAgent(observation=self._observation_config,
                                       action=self._action_config,
                                       behavior=self)
        obs = self._gym_agent.update_observations()
        act, _ = self._policy.predict(obs, deterministic=self.deterministic)
        cmd = self._gym_agent.get_cmd_from_action(act, time_step)
        return self.feasible_twist(cmd)

    @classmethod
    def clone_behavior(cls,
                       behavior: core.Behavior,
                       policy: Any,
                       action_config: ControlActionConfig,
                       observation_config: ObservationConfig,
                       deterministic: bool = False) -> 'PolicyBehavior':
        """
        Configure a new policy behavior from a behavior.

        :param      behavior:         The behavior to replicate
        :param      policy:           The policy
        :param      config:           The configuration
        :param      deterministic:    Whether or not to output deterministic actions

        :returns:   The configured policy behavior
        """
        pb = cls(policy=policy,
                 action_config=action_config,
                 observation_config=observation_config,
                 deterministic=deterministic)
        pb.set_state_from(behavior)
        return pb

    def clone(self) -> 'PolicyBehavior':
        """
        Creates a new policy behavior with same properties but a separate state.

        :returns:   Copy of this object.
        """
        return PolicyBehavior.clone_behavior(
            behavior=self,
            policy=self._policy,
            action_config=self._action_config,
            observation_config=self._observation_config,
            deterministic=self._deterministic)
