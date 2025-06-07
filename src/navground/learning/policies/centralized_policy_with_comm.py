from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import torch as th
from navground.core import FloatType
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution, StateDependentNoiseDistribution)
from stable_baselines3.common.torch_layers import (CombinedExtractor,
                                                   FlattenExtractor,
                                                   create_mlp)
from stable_baselines3.sac.policies import (LOG_STD_MAX, LOG_STD_MIN, Actor,
                                            BasePolicy, SACPolicy)
from torch import nn

if TYPE_CHECKING:

    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.type_aliases import (PyTorchObs, Schedule,
                                                       TensorDict)


class StackCombinedExtractor(CombinedExtractor):

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.stack(encoded_tensor_list, dim=-1).flatten(start_dim=1)


class ActorWithComm(Actor):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.spaces.Box,
                 net_arch: list[int],
                 features_extractor: nn.Module,
                 features_dim: int,
                 activation_fn: type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 log_std_init: float = -3,
                 full_std: bool = True,
                 use_expln: bool = False,
                 clip_mean: float = 2.0,
                 normalize_images: bool = True,
                 comm_space: gym.spaces.Box = gym.spaces.Box(low=-1,
                                                             high=1,
                                                             dtype=FloatType),
                 comm_net_arch: list[int] = [32, 32]):
        self.comm_space = comm_space
        self.comm_size = gym.spaces.utils.flatdim(comm_space)
        self.comm_net_arch = net_arch[:]
        super().__init__(observation_space,
                         action_space,
                         net_arch,
                         features_extractor,
                         features_dim=features_dim,
                         activation_fn=activation_fn,
                         use_sde=use_sde,
                         log_std_init=log_std_init,
                         full_std=full_std,
                         use_expln=use_expln,
                         clip_mean=clip_mean,
                         normalize_images=normalize_images)
        self._has_dict_space = isinstance(observation_space, gym.spaces.Dict)
        self.number = self.action_space.shape[0]
        self.single_agent_features_dim = features_dim // self.number
        features_dim = self.single_agent_features_dim + self.comm_size
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)

        tx_net = create_mlp(self.single_agent_features_dim, -1,
                            [*self.comm_net_arch, self.comm_size], nn.Sigmoid)
        self.tx_net = nn.Sequential(*tx_net)
        action_dim = self.action_space.shape[1]
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                learn_features=True,
                squash_output=True)
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init)
            if clip_mean > 0.0:
                self.mu = nn.Sequential(
                    self.mu, nn.Hardtanh(min_val=-clip_mean,
                                         max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(
                action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim,
                                     action_dim)  # type: ignore[assignment]

    def get_action_dist_params_after_extraction(
        self, features: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def scale_comm(self, values: th.Tensor) -> th.Tensor:
        # TODO(Jerome): generalize to non-homogeneous comm spaces
        low, high = self.comm_space.low[0], self.comm_space.high[0]
        return low + values * (high - low)

    def get_action_dist_params(
            self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        features = features.reshape(
            (-1, self.number, self.single_agent_features_dim))
        tx = self.tx_net(features)
        tx = self.scale_comm(tx)
        self._last_tx = tx.detach().numpy()
        rx = []
        for i in range(self.number - 1):
            rx.append(th.roll(tx, i + 1, dims=1))
        # self._last_rxs = [r.detach().numpy() for r in rx]

        features = th.concat([features, *rx], dim=-1)
        features = features.reshape((-1, features.shape[-1]))
        m, le, info = self.get_action_dist_params_after_extraction(features)
        shape = cast('gym.spaces.Box', self.action_space).shape
        s = shape[0] * shape[1]
        return m.reshape((-1, s)), le.reshape((-1, s)), info


class SACPolicyWithComm(SACPolicy):
    """
    This class describes a SB3 Sac policy that works
    on a stacked multi-agent environments.
    (:py:class:`navground.learning.parallel_env.JointEnv`)

    The actor is composed of two modules: CommNet and ActionNet.
    CommNet takes stacked (i.e. batched) single agent observations and
    computes a message (for each agent). ActionNet takes stacked (i.e. batched)
    single agent observations and all other agents messages and computes actions.

    Communication never exit the policy during training.
    During inference, we can evaluate the two sub-networks separately
    and explicitly share the messages, using :py:class:`DistributedCommPolicy`.

    The class supports composed and simple observations, selecting the corresponding
    features extractor at initialization.

    User can configure the CommNet by including fields

    - comm_space: gym.space.Box
    - comm_net_arch: list[int]

    in ``policy_kwargs``.
    """

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
        comm_space: gym.spaces.Box = gym.spaces.Box(low=-1,
                                                    high=1,
                                                    dtype=FloatType),
        comm_net_arch: list[int] = [32, 32],
    ):
        """
        Constructs a new instance.

        Same parameters of the super class :py:class:`SACPolicy` constructor,
        with the addition of ``actor_specs``.

        :param      comm_space:    The comm space.
        :param      comm_net_arch: The comm network MLP specs
        """
        self._comm_space = comm_space
        self._comm_net_arch = comm_net_arch
        if features_extractor_class is None:
            if isinstance(observation_space, gym.spaces.Dict):
                features_extractor_class = StackCombinedExtractor
            else:
                features_extractor_class = FlattenExtractor
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

    def make_actor(
            self,
            features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs,
                                                       features_extractor)
        return ActorWithComm(
            comm_space=self._comm_space,
            comm_net_arch=self._comm_net_arch,
            **actor_kwargs,
        ).to(self.device)


class DistributedCommPolicy(BasePolicy):
    """
    This class converts a :py:class:`SACPolicyWithComm` centralized policy
    in a distributed policy, evaluating the two sub-networks
    to compute action and outgoing message, which are returned concatenated.
    """

    def __init__(self, observation_space: gym.spaces.Dict | gym.spaces.Box,
                 action_space: gym.spaces.Box, policy: SACPolicyWithComm,
                 **kwargs: Any):
        """
        Constructs a new instance.

        :param      observation_space:  The observation space
        :param      action_space:       The action space
        :param      policy:             The centralized policy
        :param      kwargs:             The keywords arguments passed
                                        to the super class
        """
        super().__init__(observation_space, action_space, **kwargs)
        assert isinstance(policy.actor, ActorWithComm)
        self._actor: ActorWithComm = policy.actor
        if isinstance(observation_space, gym.spaces.Dict):
            extra_keys = set(observation_space.spaces) - set(
                cast('gym.spaces.Dict', self._actor.observation_space).spaces)
            assert len(extra_keys) == 1
            self._has_dict_space = True
            self.comm_key = extra_keys.pop()
            self.comm_extractor = nn.Flatten()

    def forward(self,
                observation: PyTorchObs,
                deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self,
                 observation: PyTorchObs,
                 deterministic: bool = False) -> th.Tensor:
        if self._has_dict_space:
            comm_observation = cast('dict[str, th.Tensor]',
                                    observation).pop(self.comm_key)
            features_without_comm = self._actor.extract_features(
                observation, self._actor.features_extractor)
            features_comm = self.comm_extractor(comm_observation).reshape(
                *features_without_comm.shape[:-1], -1)
            features = th.cat([features_without_comm, features_comm], dim=-1)
        else:
            features = cast('th.Tensor', observation)
            features_without_comm = features[
                ..., :self._actor.single_agent_features_dim]
        tx = self._actor.tx_net(features_without_comm)
        mean_actions, log_std, kwargs = self._actor.get_action_dist_params_after_extraction(
            features)
        act = self._actor.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs)
        space = cast('gym.spaces.Box', self.action_space)
        low, high = space.low[0], space.high[0]
        act = low + (0.5 * (act + 1.0) * (high - low))
        tx = self._actor.scale_comm(tx)
        return th.concat([act, tx], dim=-1)

    def set_training_mode(self, mode: bool) -> None:
        pass
