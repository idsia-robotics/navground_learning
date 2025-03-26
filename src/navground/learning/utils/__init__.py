import gc


def clean_tqdm_rich() -> None:
    for obj in gc.get_objects():
        if 'tqdm_rich' in type(obj).__name__:
            try:
                obj.close()
            except Exception:
                pass
