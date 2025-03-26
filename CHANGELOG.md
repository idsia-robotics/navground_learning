# Changelog

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
