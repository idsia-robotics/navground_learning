from __future__ import annotations

from typing import SupportsFloat, cast

from navground import core

from ...config import ControlActionConfig, DefaultObservationConfig
from ...parallel_env.join import stack_observations
from ...types import (AnyPolicyPredictor, GroupObservationsTransform,
                      Observation, ObservationTransform, PathLike)
from .base import BasePolicyMixin


def get_obs(behavior: GroupedPolicyBehavior) -> Observation:
    obs = behavior.update_observation()
    if behavior._pre:
        obs = behavior._pre(obs)
    return obs


class GroupPolicyBehavior(core.BehaviorGroup):

    def __init__(self, policy: AnyPolicyPredictor):
        core.BehaviorGroup.__init__(self)
        self.policy = policy
        self._pre: GroupObservationsTransform | None = None

    def set_pre(self, value: GroupObservationsTransform | None) -> None:
        self._pre = value

    def compute_cmds(self, time_step: SupportsFloat) -> list[core.Twist2]:
        behaviors = cast('list[GroupedPolicyBehavior]', self.members)
        obss = {i: get_obs(behavior) for i, behavior in enumerate(behaviors)}
        if self._pre:
            obss = self._pre(obss)
        obs = stack_observations(obss)
        acts, _ = self.policy.predict(obs, deterministic=True)
        acts = acts.reshape(len(self.members), -1)
        return [
            behavior.get_cmd_from_action(act, float(time_step))
            for behavior, act in zip(behaviors, acts, strict=True)
        ]


class GroupedPolicyBehavior(BasePolicyMixin,
                            core.BehaviorGroupMember,
                            name="GroupPolicy",
                            include_properties_of=[BasePolicyMixin]):

    _groups: dict[int, core.BehaviorGroup] = {}

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
        self._group_pre: GroupObservationsTransform | None = None
        BasePolicyMixin.__init__(self, policy, policy_path, action_config,
                                 observation_config, deterministic, pre)
        core.BehaviorGroupMember.__init__(self, kinematics, radius)

    def set_group_pre(self, value: GroupObservationsTransform | None) -> None:
        self._group_pre = value

    def make_group(self) -> core.BehaviorGroup:
        if self.policy_path:
            policy: AnyPolicyPredictor | None = self.load_policy(
                self.policy_path)
        else:
            policy = self._policy
        if not policy:
            raise RuntimeError("Policy not set")
        g = GroupPolicyBehavior(policy)
        g.set_pre(self._group_pre)
        return g

    def get_groups(self) -> dict[int, core.BehaviorGroup]:
        return self._groups

    def get_group_hash(self) -> int:
        if self.policy_path:
            return abs(hash(self.policy_path))
        return abs(hash(self._policy))

    def close(self) -> None:
        core.BehaviorGroupMember.close(self)

    def prepare(self) -> None:
        BasePolicyMixin.prepare(self)
        core.BehaviorGroupMember.prepare(self)
