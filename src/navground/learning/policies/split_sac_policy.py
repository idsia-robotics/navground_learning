from __future__ import annotations

import sys
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import gymnasium as gym
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import \
    SquashedDiagGaussianDistribution  # StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (CombinedExtractor,
                                                   FlattenExtractor)
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import Actor, BasePolicy, SACPolicy
from torch import nn

if TYPE_CHECKING:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

# None = all, slice for boxes, collection for dict
InputSpec = slice | Collection[str] | None
# (action_dim, input)
NetArch = list[int] | dict[str, list[int]] | None
ActorSpec = tuple[int, InputSpec, NetArch]

T = TypeVar('T', bound=gym.Space[Any])


def make_subspace(space: T, inputs: InputSpec) -> T:
    if inputs is None:
        return space
    if isinstance(inputs, slice):
        if isinstance(space, gym.spaces.Box):
            return gym.spaces.Box(  # type: ignore
                space.low[inputs], space.high[inputs])
        else:
            raise ValueError("Slice indices must be used only with box spaces")
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({  # type: ignore
            k: v
            for k, v in space.spaces.items() if k in inputs
        })
    raise ValueError("String indices must be used only with dict spaces")


def make_actors(specs: list[ActorSpec], observation_space: gym.Space,
                action_space: gym.spaces.Box, **kwargs):
    actors = []
    out_index = 0
    for (dims, inputs, net_arch) in specs:
        actor_observation_space = make_subspace(observation_space, inputs)
        actor_action_space = make_subspace(action_space,
                                           slice(out_index, out_index + dims))
        out_index += dims
        actor_kwargs = dict(**kwargs)
        actor_kwargs['features_dim'] = gym.spaces.utils.flatdim(
            actor_observation_space)
        actor_kwargs['features_extractor'] = actor_kwargs[
            'features_extractor_class'](
                actor_observation_space,
                **actor_kwargs['features_extractor_kwargs'])
        del actor_kwargs['features_extractor_class']
        del actor_kwargs['features_extractor_kwargs']
        if net_arch is not None:
            actor_kwargs['net_arch'] = net_arch
        actors.append(
            Actor(actor_observation_space, actor_action_space, **actor_kwargs))
    return actors


def filter_obs(obs, spec: InputSpec):
    if spec is None:
        return obs
    if isinstance(spec, slice):
        return obs[..., spec]
    return {k: v for k, v in obs.items() if k in spec}


class ActorList(Actor):

    def __init__(self,
                 actor_specs: list[ActorSpec],
                 observation_space: gym.Space,
                 action_space: gym.spaces.Box,
                 normalize_images: bool = True,
                 **kwargs: Any):
        BasePolicy.__init__(self,
                            observation_space,
                            action_space,
                            normalize_images=normalize_images,
                            squash_output=True)
        self.actors = make_actors(actor_specs,
                                  observation_space=observation_space,
                                  action_space=action_space,
                                  normalize_images=normalize_images,
                                  **kwargs)
        for i, actor in enumerate(self.actors):
            self.add_module(f'actor_{i}', actor)
        action_dim = get_action_dim(self.action_space)
        self.action_dist = SquashedDiagGaussianDistribution(  # type: ignore[assignment]
            action_dim)
        self.filters = [f for _, f, _ in actor_specs]

    def set_training_mode(self, mode):
        for actor in self.actors:
            actor.set_training_mode(mode)

    def get_action_dist_params(
            self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        outs: list[tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]] = [
            actor.get_action_dist_params(filter_obs(obs, f))
            for actor, f in zip(self.actors, self.filters, strict=True)
        ]
        return (th.concat([out[0] for out in outs],
                          dim=-1), th.concat([out[1] for out in outs],
                                             dim=-1), {})


def update_learning_rates(optimizer: th.optim.Optimizer,
                          learning_rates: Iterable[float]) -> None:
    for param_group, learning_rate in zip(optimizer.param_groups,
                                          learning_rates,
                                          strict=True):
        param_group["lr"] = learning_rate


class SplitSACPolicy(SACPolicy):
    """
    A SAC policy whose actor contains independent MLP
    that outputs some of the actions.

    That is, instead of the monolithic actor that takes
    all observations and computes all actions,
    the actor of this policy can use subset of the observations
    to compute some of the actions.

    The sub-modules are configured from argument :py:obj:`actor_specs` in
    ``policy_kwargs`` of type :py:type:`list[ActorSpec]`.

    The different tuples ``(action_size, input_spec, net_arch)``
    each configure one of the MLPs that computes ``action_size`` actions
    using observations specified by``input_spec`` with an architecture ``net_arch``.

    The list should be ordered and actions sizes
    should sum up to the total size of the action space.

    For example:

    >>> env.observation_space
    Dict('a': Box(0.0, 1.0, (1,), float32), 'b': Box(0.0, 1.0, (1,), float32))
    >>> env.action_space
    Box(-1.0, 1.0, (2,), float32)
    >>> actor_specs = [(1, None, [64, 64]), (1, ['a'], [16, 16])]
    >>> model = SAC(env, policy_kwargs={'actor_specs': actor_specs})

    creates a model with a policy that computes two actions:

    - the first using a 64 + 64 MLP from ``a`` and ``b``
    - the second using a 16 + 16 MLP solely from ``a``

    """

    actor: ActorList
    actor_specs: list[ActorSpec]

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] | None = None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_specs: list[ActorSpec] = [],
    ):
        """
        Constructs a new instance.

        Same parameters of the super class :py:class:`SACPolicy` constructor,
        with the addition of :py:obj:`actor_specs`.

        :param      actor_specs:  The actor specs.
        """

        if features_extractor_class is None:
            if isinstance(observation_space, gym.spaces.Box):
                features_extractor_class = FlattenExtractor
            else:
                features_extractor_class = CombinedExtractor
        self.actor_specs = actor_specs

        SACPolicy.__init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.actor.optimizer = self.optimizer_class(
            [{
                'params': actor.parameters()
            } for actor in self.actor.actors],
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs)

    def make_actor(
            self,
            features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        actor_kwargs = dict(**self.actor_kwargs)
        actor_kwargs[
            'features_extractor_class'] = self.features_extractor_class
        actor_kwargs[
            'features_extractor_kwargs'] = self.features_extractor_kwargs
        return ActorList(actor_specs=self.actor_specs,
                         **actor_kwargs).to(self.device)  # type: ignore


def setup_actor_learning_rates(model: SAC,
                               learning_rates: list[Schedule | float]) -> None:
    """
    Set the learning rates of the sub-modules composing the actor.

    Only applied if the model policy is of class :py:class:`SplitSACPolicy`.

    :param      model:           The model

    :param   learning_rates:  A list of tuples
        ``(number of training steps, learning rate)``,
        one for each sub-module of the actor.
    """

    if isinstance(model.policy, SplitSACPolicy):
        print("Model policy type is not SplitSACPolicy", file=sys.stderr)
        return

    lr_schedules = [get_schedule_fn(lr) for lr in learning_rates]

    def update_learning_rate(
            optimizers: list[th.optim.Optimizer] | th.optim.Optimizer) -> None:
        lrs = [fn(model._current_progress_remaining) for fn in lr_schedules]
        for i, lr in enumerate(lrs):
            model.logger.record(f"train/learning_rate_actor{i}", lr)
        update_learning_rates(model.actor.optimizer, lrs)
        if isinstance(optimizers, list):
            SAC._update_learning_rate(model, optimizers[1:])

    model._update_learning_rate = update_learning_rate  # type: ignore[method-assign]


class AlternateActorCallback(BaseCallback):
    """
    A callback that alternates training modules that
    compose the actor of :py:class:`SplitSACPolicy`.
    """

    def __init__(self,
                 learning_rates: list[tuple[int, Schedule | float]],
                 exclusive: bool = True,
                 verbose: int = 0):
        """
        Constructs a new instance.

        :param      learning_rates:  The learning rates, as a list of
            tuple (number of training steps, learning rate),
            one for each sub-module of the :py:class:`SplitSACPolicy`
            actor.
        :param      exclusive: Whether to set the learning rate to the
                other modules (``exclusive=False) or to remove their
                parameters from the optimization (``exclusive=True)

        :param      verbose:         The verbosity level
        """
        super().__init__(verbose)
        self._next_actor_index = 0
        self._next_actor_step = 0
        self._learning_rates = learning_rates
        self._exclusive = exclusive

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_actor_step:
            # print(self.model.policy.actor.optimizer)
            steps, lr = self._learning_rates[self._next_actor_index]
            self.logger.record("train/actor_index", self._next_actor_index)
            self._next_actor_step = self.num_timesteps + steps
            if self._exclusive:
                actor = cast('ActorList', self.model.policy.actor)
                actor.optimizer = self.model.policy.optimizer_class(
                    actor.actors[self._next_actor_index].parameters(),
                    lr=lr,  # type: ignore[call-arg]
                    **self.model.policy.optimizer_kwargs)
                self.model.lr_schedule = get_schedule_fn(lr)
            else:
                learning_rates = [
                    lr if i == self._next_actor_index else 0
                    for i, (_, lr) in enumerate(self._learning_rates)
                ]
                setup_actor_learning_rates(cast('SAC', self.model),
                                           learning_rates)
            self._next_actor_index = (self._next_actor_index + 1) % len(
                self._learning_rates)
        return True
