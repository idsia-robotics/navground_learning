def test_import():
    import navground.learning.utils.benchmarl  # noqa


def test_create_env():
    from navground.learning.examples.cross import get_env
    from navground.learning.utils.benchmarl import make_env

    penv = get_env(multi_agent=True)
    env = make_env(env=penv, seed=0, categorical_actions=False)
    env.set_seed(0)
    env.reset()
