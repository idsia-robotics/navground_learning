import numpy as np
from imitation.data.types import TrajectoryWithRew, maybe_wrap_in_dictobs
from navground import sim


def get_trajectories_from_run(run: sim.ExperimentalRun):
    trajectories = []
    for key, acts in run.get_records('actions').items():
        ds = run.get_records(f'observations/{key}')
        obs = {k: np.asarray(v) for k, v in ds.items()}
        dict_obs = maybe_wrap_in_dictobs(obs)
        rews = run.get_record(f'rewards/{key}')
        if rews is None:
            rews = np.zeros(len(acts))
        trajectory = TrajectoryWithRew(obs=dict_obs,
                                       acts=np.asarray(acts),
                                       rews=np.asarray(rews),
                                       infos=None,
                                       terminal=True)
        trajectories.append(trajectory)
    return trajectories


def get_trajectories_from_experiment(
        experiment: sim.Experiment) -> list[TrajectoryWithRew]:
    ts = [get_trajectories_from_run(run) for _, run in experiment.runs.items()]
    return sum(ts, [])
