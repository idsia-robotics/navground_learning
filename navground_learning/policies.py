from typing import Callable, Collection, Dict, Sequence

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   NatureCNN, create_mlp,
                                                   get_flattened_obs_dim,
                                                   is_image_space)

# TODO: make sure images are not grouped.


class OrderInvariantCombinedExtractor(BaseFeaturesExtractor):
    """

    A variation of SB3 `CombinedExtractor`
    that applies a ordering invariant MLP feature extractor to a group of keys
    after optionally masking it.

    Same as the original `CombinedExtractor`:
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).

    :param order_invariant_keys: the keys to group together and process by an ordering invariant feature extractor
    :param filter_key: the key to use for masking to select items with positive values of this key
    :param net_arch: the ordering invariant MLP layers sizes
    :param activation_fn: the ordering invariant MLP activation function
    """

    def __init__(self,
                 observation_space: spaces.Dict,
                 cnn_output_dim: int = 256,
                 normalized_image: bool = False,
                 order_invariant_keys: Collection[str] = [],
                 filter_key="",
                 net_arch: Sequence[int] = [8],
                 activation_fn=nn.ReLU,
                 reductions: Sequence[Callable[[th.Tensor, int, bool],
                                               th.Tensor]] = [th.sum]):

        super().__init__(observation_space, features_dim=1)

        # Same as CombinedExtractor
        extractors: Dict[str, nn.Module] = {}
        order_invariant_lens = set()
        total_concat_size = 0
        order_invariant_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace,
                                            features_dim=cnn_output_dim,
                                            normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                if key in order_invariant_keys:
                    order_invariant_lens.add(subspace.shape[0])
                    order_invariant_size += get_flattened_obs_dim(subspace)
                elif key == filter_key:
                    order_invariant_lens.add(subspace.shape[0])
                else:
                    total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict({
            k: v
            for k, v in extractors.items()
            if k not in order_invariant_keys and k != filter_key
        })
        self.order_invariant_extractors = nn.ModuleDict(
            {k: v
             for k, v in extractors.items() if k in order_invariant_keys})
        self.filter_key = filter_key
        assert len(order_invariant_lens) == 1
        order_invariant_len = order_invariant_lens.pop()
        order_invariant_size = order_invariant_size // order_invariant_len
        self._order_invariant_shape = (order_invariant_size,
                                       order_invariant_len)
        self.net_out_dim = net_arch[-1] * len(reductions)
        self._features_dim = total_concat_size + self.net_out_dim
        mlp = create_mlp(order_invariant_size, 0, net_arch, activation_fn)
        self.nn = nn.Sequential(*mlp)
        self.reductions = reductions

    def forward(self, observations: th.Tensor) -> th.Tensor:
        order_invariant_encoded_tensor_list = []
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        if self.order_invariant_extractors:

            for key, extractor in self.order_invariant_extractors.items():
                order_invariant_encoded_tensor_list.append(
                    extractor(observations[key]))
            order_invariant_input = th.cat(
                order_invariant_encoded_tensor_list,
                dim=1).reshape(self._order_invariant_shape).t()

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
