from __future__ import annotations

from typing import cast

from navground import core

from ...config import ControlActionConfig, DefaultObservationConfig
from ...parallel_env.join import stack_observations
from ...types import AnyPolicyPredictor, ObservationTransform, PathLike
from .base import BasePolicyMixin


class GroupPolicyBehavior(core.BehaviorGroup):

    def __init__(self, policy: AnyPolicyPredictor):
        super().__init__()
        self.policy = policy

    def compute_cmds(self, time_step: float) -> list[core.Twist2]:

        behaviors = cast('list[GroupedPolicyBehavior]', self.members)

        obs = stack_observations({
            i: behavior.update_observation()
            for i, behavior in enumerate(behaviors)
        })
        acts, _ = self.policy.predict(obs, deterministic=True)
        acts = acts.reshape(len(self.members), -1)
        return [
            behavior.get_cmd_from_action(act, time_step)
            for behavior, act in zip(behaviors, acts, strict=True)
        ]


class GroupedPolicyBehavior(BasePolicyMixin,
                            core.BehaviorGroupMember,
                            name="GroupPolicy",
                            include_properties_of=[BasePolicyMixin]):

    _groups: dict[int, GroupPolicyBehavior] = {}

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
        BasePolicyMixin.__init__(self, policy, policy_path, action_config,
                                 observation_config, deterministic, pre)
        core.BehaviorGroupMember.__init__(self, kinematics, radius)

    def make_group(self) -> core.BehaviorGroup:
        if self.policy_path:
            policy: AnyPolicyPredictor | None = self.load_policy(self.policy_path)
        else:
            policy = self._policy
        if not policy:
            raise RuntimeError("Policy not set")
        return GroupPolicyBehavior(policy)

    def get_groups(self) -> dict[int, GroupPolicyBehavior]:
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
