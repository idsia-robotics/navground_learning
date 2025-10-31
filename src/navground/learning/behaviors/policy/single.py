from __future__ import annotations

import warnings
from typing import SupportsFloat

from navground import core

from ...config import ControlActionConfig, DefaultObservationConfig
from ...types import AnyPolicyPredictor, ObservationTransform, PathLike
from .base import BasePolicyMixin


class PolicyBehavior(BasePolicyMixin, core.Behavior, name="Policy",
                     include_properties_of=[BasePolicyMixin]):
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
    :param action_config: Configures which actions the policy generates
    :param observation_config: Configures which observations the policy consumes
    :param deterministic: Whether the policy evaluation is deterministic
    :param pre: Optional input (observations) transformation
    """

    def __init__(self,
                 kinematics: core.Kinematics | None = None,
                 radius: float = 0.0,
                 policy: AnyPolicyPredictor | None = None,
                 policy_path: PathLike = '',
                 action_config: ControlActionConfig = ControlActionConfig(),
                 observation_config:
                 DefaultObservationConfig = DefaultObservationConfig(),
                 deterministic: bool = False,
                 pre: ObservationTransform | None = None):
        core.Behavior.__init__(self, kinematics, radius)
        BasePolicyMixin.__init__(self, policy, policy_path, action_config,
                                 observation_config, deterministic, pre)

    def prepare(self) -> None:
        BasePolicyMixin.prepare(self)
        if not self._policy:
            self.init_policy()

    def compute_cmd_internal(self, time_step: SupportsFloat) -> core.Twist2:
        # if self.check_if_target_satisfied():
        #     return core.Twist2((0, 0), 0, frame=core.Frame.relative)
        time_step = float(time_step)
        self.prepare()
        if not self._policy:
            warnings.warn("Policy not set", stacklevel=1)
            return core.Twist2((0, 0), 0, frame=core.Frame.relative)
        obs = self.update_observation()
        if self._pre:
            obs = self._pre(obs)
        act, _ = self._policy.predict(obs, deterministic=self.deterministic)
        return self.get_cmd_from_action(act, time_step)
