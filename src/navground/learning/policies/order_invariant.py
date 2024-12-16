from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from typing import TypeAlias, cast

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import (get_flattened_obs_dim,
                                                    is_image_space)
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   NatureCNN, create_mlp)
from torch.masked import as_masked_tensor

Reduction: TypeAlias = Callable[[th.Tensor, int, bool], th.Tensor]

# TODO: make sure images are not grouped.


class OrderInvariantCombinedExtractor(BaseFeaturesExtractor):
    """
    A variation of SB3 :py:class:`stable_baselines3.common.torch_layers.CombinedExtractor`
    that applies a ordering invariant MLP feature extractor to a group of keys
    after optionally masking it.

    Same as the original ``CombinedExtractor``:
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space: the observation space
    :param cnn_output_dim: Number of features to output from each CNN submodule(s).
                           Defaults to 256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).

    :param order_invariant_keys: the keys to group together and process by an
                                 ordering invariant feature extractor
    :param replicated_keys:  additional keys to add to the ordering invariant groups,
                             replicating the values for each group
    :param filter_key: the key to use for masking to select items with positive values of this key
    :param removed_keys: keys removed from the observations before concatenating
                         with ordering invariant features
    :param net_arch: the ordering invariant MLP layers sizes
    :param activation_fn: the ordering invariant MLP activation function
        If not set, it defaults to ``torch.nn.ReLU``.
    :param reductions: A sequence of (order-invariant) modules.
        If not set, it defaults to ``[torch.sum]``.

    """

    def __init__(self,
                 observation_space: gym.spaces.Dict,
                 cnn_output_dim: int = 256,
                 normalized_image: bool = False,
                 order_invariant_keys: Collection[str] = [],
                 replicated_keys: Collection[str] = [],
                 filter_key: str = "",
                 removed_keys: Collection[str] = [],
                 net_arch: list[int] = [8],
                 activation_fn: type[nn.Module] | None = None,
                 reductions: Sequence[Reduction] | None = None):

        super().__init__(observation_space, features_dim=1)
        if activation_fn is None:
            activation_fn = nn.ReLU
        if reductions is None:
            reductions = [th.sum]
        # Same as CombinedExtractor
        extractors: dict[str, nn.Module] = {}
        order_invariant_lens: set[int] = set()
        total_concat_size = 0
        order_invariant_size = 0
        replicated_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace,
                                            features_dim=cnn_output_dim,
                                            normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                if key in order_invariant_keys:
                    order_invariant_lens.add(
                        cast(gym.spaces.Box, subspace).shape[0])
                    order_invariant_size += get_flattened_obs_dim(subspace)
                elif key in replicated_keys:
                    replicated_size += get_flattened_obs_dim(subspace)
                elif key == filter_key:
                    order_invariant_lens.add(
                        cast(gym.spaces.Box, subspace).shape[0])
                elif key not in removed_keys:
                    total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict({
            k: v
            for k, v in extractors.items() if k not in order_invariant_keys
            and k != filter_key and k not in removed_keys
        })
        self.order_invariant_extractors = nn.ModuleDict({
            k: v
            for k, v in extractors.items() if k in order_invariant_keys
        })
        self.replicated_extractors = nn.ModuleDict({
            k: v
            for k, v in extractors.items() if k in replicated_keys
        })
        self.filter_key = filter_key
        assert len(order_invariant_lens) == 1
        self._number = order_invariant_lens.pop()
        order_invariant_size = order_invariant_size // self._number + replicated_size
        self.net_out_dim = net_arch[-1] * len(reductions)
        self._features_dim = total_concat_size + self.net_out_dim
        mlp = create_mlp(order_invariant_size, 0, net_arch, activation_fn)
        self.nn = nn.Sequential(*mlp)
        self.reductions = reductions

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        if self.order_invariant_extractors:
            oit = [
                extractor(observations[key]).reshape(self._number, -1)
                for key, extractor in self.order_invariant_extractors.items()
            ]
            rt = [
                extractor(observations[key]).repeat(self._number, 1)
                for key, extractor in self.replicated_extractors.items()
            ]
            order_invariant_input = th.cat(oit + rt, dim=1)
            if self.filter_key:
                mask = observations[self.filter_key].flatten() > 0
                order_invariant_input = order_invariant_input[mask]
            z = self.nn(order_invariant_input)
            if len(z):
                for reduction in self.reductions:
                    r, *_ = reduction(z, 0, True)
                    encoded_tensor_list.append(r.reshape(1, -1))
            else:
                encoded_tensor_list.append(th.zeros(1, self.net_out_dim))
        return th.cat(encoded_tensor_list, dim=1)


class OrderInvariantFlattenExtractor(BaseFeaturesExtractor):
    """

    Similar to :py:class:`OrderInvariantCombinedExtractor`
    but for flat observation spaces.

    :param observation_space: the observation space
    :param order_invariant_slices: the slices to group together and process by
                                   an ordering invariant feature extractor
    :param replicated_slices: additional slices to add to the ordering invariant groups,
                              replicating the values for each group
    :param filter_slice: the slice to use for masking to select items with
                         positive values for indices in this slice
    :param removed_slices: keys removed from the observations before concatenating
                           with ordering invariant features
    :param net_arch: the ordering invariant MLP layers sizes
    :param activation_fn: the ordering invariant MLP activation function
        If not set, it defaults to ``torch.nn.ReLU``.
    :param reductions: A sequence of (order-invariant) modules.
        If not set, it defaults to ``[torch.sum]``.
    :param use_masked_tensors: Whether to use masked tensors
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 order_invariant_slices: Collection[slice] = [],
                 replicated_slices: Collection[slice] = [],
                 filter_slice: slice | None = None,
                 removed_slices: Collection[slice] = [],
                 number: int = 0,
                 net_arch: list[int] = [8],
                 activation_fn: type[nn.Module] | None = None,
                 reductions: Sequence[Reduction] | None = None,
                 use_masked_tensors: bool = False):

        super().__init__(observation_space, features_dim=1)
        if activation_fn is None:
            activation_fn = nn.ReLU
        if reductions is None:
            reductions = [th.sum]
        self.flatten = nn.Flatten()
        input_size = math.prod(observation_space.shape)
        self._invariant_slices = order_invariant_slices
        self._filter_slice = filter_slice
        self._number = number
        size = sum(s.stop - s.start for s in order_invariant_slices)
        self._non_invariant_mask = np.ones(input_size, dtype=bool)
        for s in order_invariant_slices:
            self._non_invariant_mask[s] = False
        for s in removed_slices:
            self._non_invariant_mask[s] = False
        if filter_slice:
            self._non_invariant_mask[filter_slice] = False
        self._replicated_mask = np.zeros(input_size, dtype=bool)
        for s in replicated_slices:
            self._replicated_mask[s] = True
        self._should_replicate = any(self._replicated_mask)
        total_concat_size = sum(self._non_invariant_mask)
        order_invariant_size = size // number + sum(self._replicated_mask)
        self.net_out_dim = net_arch[-1] * len(reductions)
        self._features_dim = total_concat_size + self.net_out_dim
        mlp = create_mlp(order_invariant_size, 0, net_arch, activation_fn)
        self.nn = nn.Sequential(*mlp)
        self.reductions = reductions
        self.use_masked_tensors = use_masked_tensors

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = self.flatten(observations)
        shape = observations.shape
        ni = observations[..., self._non_invariant_mask]
        if self._invariant_slices:
            if self._filter_slice:
                mask = (observations[..., self._filter_slice] > 0).detach()
                numbers = th.sum(mask, -1)
                has_same_number = len(th.unique(numbers)) == 1
            else:
                has_same_number = True
                mask = None

            ois = [
                observations[..., s].reshape(*shape[:-1], self._number, -1)
                for s in self._invariant_slices
            ]
            if self._should_replicate:
                ri = [1] * (len(shape) + 1)
                ri[-2] = self._number
                r = observations[..., self._replicated_mask].reshape(
                    *shape[:-1], 1, -1)
                ois.append(r.repeat(*ri))
            oi = th.cat(ois, dim=-1)
            if mask is not None and not self.use_masked_tensors:
                oi = oi[mask]
                if has_same_number:
                    number = cast(int, numbers[0].item())
                    if number > 0:
                        oi = oi.reshape(*shape[:-1], number, -1)

            zs = [ni]
            if len(oi):
                z = self.nn(oi)
                if self.use_masked_tensors and mask is not None:
                    ri = [1] * len(z.shape)
                    ri[-1] = z.shape[-1]
                    mask = mask.unsqueeze(-1).repeat(*ri)
                    z = as_masked_tensor(z,
                                         mask)  # type: ignore[no-untyped-call]
                    for reduction in self.reductions:
                        r = reduction(z, -2, False)
                        zs.append(r)
                        # TODO(Jerome): check after linting
                        # zs.append(r.to_tensor(0))
                elif has_same_number:
                    for reduction in self.reductions:
                        r = reduction(z, -2, False)
                        # TODO(Jerome): if we restrict to amax/amin, no need to
                        # check output
                        if isinstance(r, th.Tensor):
                            zs.append(r)
                        else:
                            zs.append(r.values)
                else:
                    i = 0
                    rs = []
                    for tn in numbers:
                        n = tn.item()
                        if n:
                            bs = []
                            mz = z[i:i + n]
                            for reduction in self.reductions:
                                r = reduction(mz, -2, False)
                                if isinstance(r, th.Tensor):
                                    bs.append(r)
                                else:
                                    bs.append(r.values)
                            rs.append(th.concat(bs))
                        else:
                            rs.append(th.zeros(self.net_out_dim))
                            # rs.extend([th.zeros(z.shape[1:])] * len(self.reductions))
                        i += n
                    zs.append(th.stack(rs))
            else:
                zs.append(th.zeros(*shape[:-1], self.net_out_dim))
            return th.cat(zs, dim=-1)
        return ni


def make_order_invariant_flatten_extractor(
        observation_space: gym.spaces.Box,
        dict_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        order_invariant_keys: Collection[str] = [],
        replicated_keys: Collection[str] = [],
        filter_key: str = "",
        removed_keys: Collection[str] = [],
        net_arch: list[int] = [8],
        activation_fn: type[nn.Module] | None = None,
        reductions: Sequence[Reduction] | None = None,
        use_masked_tensors: bool = False) -> OrderInvariantFlattenExtractor:
    """

    Helper function that creates a :py:class:`OrderInvariantFlattenExtractor`
    using information from a dictionary space to infer the layout of the
    observation space.

    :param observation_space: the observation space
    :param      dict_space:            The dictionary space
    :param cnn_output_dim: Number of features to output from each CNN submodule(s).
                           Defaults to 256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).

    :param order_invariant_keys: the keys to group together and process by an
                                 ordering invariant feature extractor
    :param replicated_keys:  additional keys to add to the ordering invariant groups,
                             replicating the values for each group
    :param filter_key: the key to use for masking to select items with positive values of this key
    :param removed_keys: keys removed from the observations before concatenating
                         with ordering invariant features
    :param net_arch: the ordering invariant MLP layers sizes
    :param activation_fn: the ordering invariant MLP activation function
        If not set, it defaults to ``torch.nn.ReLU``.
    :param reductions: A sequence of (order-invariant) modules.
        If not set, it defaults to ``[torch.sum]``.
    :param use_masked_tensors:   Whether to use masked tensors

    :returns:   The order invariant flatten extractor.

    :raises     AssertionError:  if ``filter_key`` is associated
        to a space that is non-flat.
    """

    ns = set(
        cast(gym.spaces.Box, dict_space[k]).shape[0]
        for k in order_invariant_keys)
    if filter_key and filter_key in dict_space:
        ns.add(cast(gym.spaces.Box, dict_space[filter_key]).shape[0])
    assert len(ns) == 1
    number = list(ns)[0]
    indices = {}
    i = 0
    for k, v in dict_space.items():
        dim = gym.spaces.flatdim(v)
        indices[k] = slice(i, i + dim, 1)
        i += dim
    invariant_slices = [
        indices[k] for k in order_invariant_keys if k in indices
    ]
    removed_slices = [indices[k] for k in removed_keys if k in indices]
    replicated_slices = [indices[k] for k in replicated_keys if k in indices]
    filter_slice = indices.get(filter_key)
    return OrderInvariantFlattenExtractor(
        observation_space=observation_space,
        order_invariant_slices=invariant_slices,
        replicated_slices=replicated_slices,
        filter_slice=filter_slice,
        removed_slices=removed_slices,
        number=number,
        net_arch=net_arch,
        activation_fn=activation_fn,
        reductions=reductions,
        use_masked_tensors=use_masked_tensors)
