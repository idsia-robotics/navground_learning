def test_import():
    import navground.learning.env  # noqa


def test_create_env():
    from navground.learning.examples.corridor_with_obstacle import get_env

    env = get_env()
    env.reset(seed=0)


def test_load_env():
    from navground.learning.io import load_env
    import pathlib

    path = pathlib.Path(__file__).parent / 'env.yaml'
    env = load_env(path)
    env.reset(seed=0)
