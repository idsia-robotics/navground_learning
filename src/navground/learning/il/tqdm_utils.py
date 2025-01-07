from __future__ import annotations

import tqdm.auto


def setup_tqdm() -> None:
    """
    Disables the progress while saving the DAgger datasets
    and configures tqdm to use ``auto``.

    :returns:   { description_of_the_return_value }
    :rtype:     None
    """
    try:
        import datasets.utils  # type: ignore[import-untyped]

        datasets.utils.tqdm = tqdm.auto.tqdm
        datasets.utils.disable_progress_bar()
    except ImportError:
        pass
    try:
        import imitation.algorithms.bc

        imitation.algorithms.bc.tqdm = tqdm.auto  # type: ignore[attr-defined]
    except ImportError:
        pass
