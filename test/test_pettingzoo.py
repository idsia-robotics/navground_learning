def test_import():
    import navground.learning.parallel_env  # noqa


def test_create_env():
    from navground.learning.examples.cross import get_env

    penv = get_env(multi_agent=True)
    penv.reset(seed=0)


def test_load_env():
    from navground.learning.io import load_env
    import pathlib

    path = pathlib.Path(__file__).parent / 'parallel_env.yaml'
    penv = load_env(path)
    penv.reset(seed=0)
