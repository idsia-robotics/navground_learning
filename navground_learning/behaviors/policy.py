import logging
import pathlib
from typing import Any, Dict, Optional

from navground import core

from ..utils import GymAgent, GymAgentConfig


class PolicyBehavior(core.Behavior, name="Policy"):

    """
    A navigation behavior that evaluates a ML policy

    *Registered properties*:

    - :py:attr:`policy_path` (str)
    - :py:attr:`flat` (bool)
    - :py:attr:`history` (int)
    - :py:attr:`fix_orientation` (bool)
    - :py:attr:`include_target_direction` (bool)
    - :py:attr:`include_target_distance` (bool)
    - :py:attr:`include_velocity` (bool)
    - :py:attr:`include_radius` (bool)

    *State*: :py:class:`SensingState`

    :param kinematics: The agent kinematics
    :param radius: The agent radius
    :param policy: The policy
    :param config: How to use the policy (default if not specified)
    """

    _policies: Dict[str, Any] = {}

    def __init__(self,
                 kinematics: Optional[core.Kinematics] = None,
                 radius: float = 0.0,
                 policy: Any = None,
                 config: GymAgentConfig | None = None):
        super().__init__(kinematics, radius)
        self._state = core.SensingState()
        self._policy = policy
        self._policy_path = ""
        self._gym_agent: Optional[GymAgent] = None
        self._config = config or GymAgentConfig()

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
    @core.register(True, "Whether to include the target distance in the observations")
    def include_target_distance(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_distance`
        """
        return self._config.include_target_distance

    @include_target_distance.setter  # type: ignore[no-redef]
    def include_target_distance(self, value: bool) -> None:
        self._config.include_target_distance = value

    @property
    @core.register(True, "Whether to include the target direction in the observations")
    def include_target_direction(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_target_direction`
        """
        return self._config.include_target_direction

    @include_target_direction.setter  # type: ignore[no-redef]
    def include_target_direction(self, value: bool) -> None:
        self._config.include_target_direction = value

    @property
    @core.register(False, "Whether to include the current velocity in the observations")
    def include_velocity(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_velocity`
        """
        return self._config.include_velocity

    @include_velocity.setter  # type: ignore[no-redef]
    def include_velocity(self, value: bool) -> None:
        self._config.include_velocity = value

    @property
    @core.register(False, "Whether to include the own radius in the observations")
    def include_radius(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.include_radius`
        """
        return self._config.include_radius

    @include_radius.setter  # type: ignore[no-redef]
    def include_radius(self, value: bool) -> None:
        self._config.include_radius = value

    @property
    @core.register(False, "Whether to flatten the observations")
    def flat(self) -> bool:
        """
        See :py:attr:`GymAgentConfig.flat`
        """
        return self._config.flat

    @flat.setter  # type: ignore[no-redef]
    def flat(self, value: bool) -> None:
        self._config.flat = value

    @property
    @core.register(False, "Length of the queue")
    def history(self) -> int:
        """
        See :py:attr:`GymAgentConfig.history`
        """
        return self._config.history

    @history.setter  # type: ignore[no-redef]
    def history(self, value: int) -> None:
        self._config.history = value

    @property
    @core.register(False, "Whether to keep orientation fixed")
    def fix_orientation(self) -> int:
        """
        See :py:attr:`GymAgentConfig.fix_orientation`
        """
        return self._config.fix_orientation

    @fix_orientation.setter  # type: ignore[no-redef]
    def fix_orientation(self, value: int) -> None:
        self._config.fix_orientation = value

    def get_environment_state(self) -> core.SensingState:
        return self._state

    def cmd_twist_towards_point(self, point: core.Vector2, speed: float,
                                time_step: float,
                                frame: core.Frame) -> core.Twist2:
        if self._policy is None:
            return core.Twist2((0, 0), frame)
        if not self._gym_agent:
            self._gym_agent = GymAgent(
                self._config.configure(behavior=self, state=self._state))
        # TODO(Jerome): use point instead of target
        obs = self._gym_agent.update_observations(self, self._state)
        act, _ = self._policy.predict(obs)
        cmd = self._config.get_cmd_from_action(act)
        return self.feasible_twist(cmd, frame)

    @classmethod
    def clone_behavior(cls, behavior: core.Behavior, policy: Any,
                       config: GymAgentConfig) -> 'PolicyBehavior':
        """
        Configure a new policy behavior from a behavior.

        :param      behavior:  The behavior to replicate
        :param      policy:    The policy
        :param      config:    The configuration

        :returns:   The configured policy behavior
        """
        pb = cls(policy=policy, config=config)
        pb.set_state_from(behavior)
        return pb

    def clone(self) -> 'PolicyBehavior':
        """
        Creates a new policy behavior with same properties but a separate state.

        :returns:   Copy of this object.
        """
        return PolicyBehavior.clone_behavior(self, self._policy, self._config)
