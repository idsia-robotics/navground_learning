from __future__ import annotations

import yaml
from navground import core
from collections.abc import Collection


class SelectionBehavior(core.Behavior, name="Selection"):

    def __init__(self,
                 kinematics: core.Kinematics | None = None,
                 radius: float = 0.0,
                 behaviors: Collection[core.Behavior] = tuple()):
        super().__init__(kinematics, radius)
        self.behaviors = list(behaviors)
        self._state = core.GeometricState()
        self._index = 0

    def decode(self, value: str) -> None:
        data = yaml.safe_load(value)
        behaviors = data.get('behaviors', [])
        for behavior in behaviors:
            b = core.load_behavior(yaml.safe_dump(behavior))
            if b:
                self.behaviors.append(b)

    def encode(self) -> str:
        behaviors = [
            yaml.safe_load(core.dump(behavior)) for behavior in self.behaviors
        ]
        if behaviors:
            return yaml.safe_dump({'behaviors': behaviors})
        return ''

    @property
    @core.register(0, "The index of the active behavior")
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        self._index = max(0, value)

    def compute_cmd_internal(self, time_step: float) -> core.Twist2:
        if self.index < len(self.behaviors):
            behavior = self.behaviors[self.index]
            behavior.set_state_from(self)
            state = behavior.environment_state
            if isinstance(state, core.GeometricState):
                state.neighbors = self._state.neighbors
                state.static_obstacles = self._state.static_obstacles
                state.line_obstacles = self._state.line_obstacles
            return behavior.compute_cmd_internal(time_step)
        return super().compute_cmd_internal(time_step)

    def get_environment_state(self) -> core.EnvironmentState:
        return self._state
