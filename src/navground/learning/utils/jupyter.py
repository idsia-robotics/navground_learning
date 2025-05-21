"""
Modified from https://github.com/HerveMignot/switch_magic
MIT Licence, Copyright (c) 2018 Hervé Mignot
"""

from __future__ import annotations
import gc

from IPython.core.magic import register_cell_magic
from IPython import get_ipython


@register_cell_magic
def skip_if(line: str, cell: str) -> None:
    """
    Executes the cell if the condition in ``line`` evaluate to False.

    See `switch_magic <https://github.com/HerveMignot/switch_magic>`_.

    :param line:  The condition
    :param cell:  The cell
    """
    if eval(line):
        return
    get_ipython().run_cell(cell)


@register_cell_magic
def run_if(line: str, cell: str) -> None:
    """
    Executes the cell if the condition in ``line`` evaluate to True.

    See `switch_magic <https://github.com/HerveMignot/switch_magic>`_.

    :param line:  The condition
    :param cell:  The cell
    """
    if not eval(line):
        return
    get_ipython().run_cell(cell)


@register_cell_magic
def run_and_time_if(line: str, cell: str) -> None:
    """
    Executes the cell if the condition in ``line`` evaluate to True
    and time it.

    See `switch_magic <https://github.com/HerveMignot/switch_magic>`_.

    :param line:  The condition
    :param cell:  The cell
    """
    if not eval(line):
        return
    get_ipython().magics_manager.magics['cell']['time'](cell=cell)


def clean_tqdm_rich() -> None:
    """
    Force closing any tqdm rich object
    """
    for obj in gc.get_objects():
        if 'tqdm_rich' in type(obj).__name__:
            try:
                obj.close()
            except Exception:
                pass
