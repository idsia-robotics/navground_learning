# Changelog

## [0.2] 2025-08-06

We release several significant additions, such as support for:

- TorchRL and BenchMARL: several SoA MARL algorithms that support heterogeneous groups too.

- Success/failure criteria: a significant number of navigation tasks (e.g., "move safely to a location") require/offer binary outcome.

- Global state to increase training stability in MARL tasks.

- Learning to navigation *and* communicate is a very interesting problem, which we can now tackle using `navground-learning`. We added a (quite long) tutorials analyzing a prototypical task around navigation+communication. 

### Added

- Support for multiple sensors, which have been supported by navground since v0.4.

- Success/failure criteria.
	- Success recording in evaluation experiments (and logger).

- Global state in multi-agent environments.

- Support for TorchRL and BenchMARL.
	- tutorial on TorchRL integration.
	- wrapper `NameWrapper` to index agent by string, like agent_0, agent_1, ... .

- Support to train centralized policies
	- `JointEnv` stacks actions and observation from a parallel environment as a single-agent environment, aggregating rewards, terminations and truncations, which is useful to train centralized policies. 

- Support to train policy that exchange messages.
	- Probe to record transmitted messages.
	- State estimation to exchange messages.
	- SB3 SAC policy to train a distributed policy with communication centrally.
	- Control action configurations for agents that exchange messages.

- Discrete actions (multi-binary and discrete spaces).
	- `OnnxPolicy` accepts now an action_space to support discrete actions (whose limits get lost in the conversion).

- Policies that uses uses multiple MLP (`SplitMLP`) for SB3 and BenchMARL.

- Behavior `GroupedPolicyBehavior` that evaluates a policy for a *group* of agents at once.

- Options to specify how `DefaultObservationConfig` should handle dict spaces: flatten sub-spaces, ignore some keys, sort the keys, normalize values. 

- Optional position and orientation observations in `DefaultObservationConfig`.

- Parallel environment wrappers
	- `NameWrapper` to index agent by string, like agent_0, agent_1, ... .
	- `MaskWrapper` to mask part of the action and observation spaces.

- `FixedReward` (to optimize duration) and `TargetEfficacyReward` (to optimize efficacy while moving towards a pose).

- `Pad` scenario a very simple yet rich and challenging scenario where two agents cross along a corridor and needs to coordinate their passing over a pad: scenario, example, lots of tutorials, and model-based policies.

- We can now save videos and plots while logging.

- SB3 callbacks 
	- Exporting the best model to onnx
	- Enhanced progress bar  that display mean reward and success (similar to BenchMARL).
	- Recording a video during evaluation.

- Helpers for:
	- displaying/recording videos during evaluation.
	- for working in `jupyter` (used in the tutorials).
	- plotting logs and policies
	- performing rollouts, similar to TorchRL `env.rollout`.

- example directory with helper functions used in the tutorials.

- "go to pose" tutorial.

### Fixed

- vectorized envs created using `make_vec_from_penv` now have a complete setter/getter chain (added missing `get_attr`, `set_attr` from `SuperSuit`).

### Changed

- set minimal Navground version to >= 0.6.
- renamed `terminate_outside_bounds` to `truncate_outside_bounds`.
- changed typing of `onnx.export` to use a protocol (vs `BasePolicy` from SB3).
- `make_vec_from_penv` now accepts a seed.


## [0.1.1] 2025-26-03

Patch release with few fixes.

### Fixed

- Gymnasium environments now set action and observation spaces correctly after loading from dict/YAML.
- Fixed `PolicyBehavior` properties
- Fixed the configurations `__repr__` method.

### Changed

- `OnnxPolicy` now uses a single thread.
- Postponed `PolicyBehavior` policy loading to the first evaluation.
- `PolicyBehavior.policy_path` now always returns a relative path

## [0.1.0] 2024-16-12

First official release.
