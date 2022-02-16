import json

from diora import DioraMLP
from hard_diora import DioraMLPWithTopk
from experiment_logger import get_logger


name_to_clz = {
    'mlp': DioraMLP,
    'topk-mlp': DioraMLPWithTopk,
}


def get_diora(options, context, config):
    config = json.loads(config)

    assert isinstance(config, dict), "Config with value {} is not type dict.".format(config)

    assert len(config.keys()) == 1, "Config should have 1 key only."

    name = list(config.keys())[0]

    # Use default config if it exists.
    kwargs_dict = get_default_configs(options, context).get(name, {})
    # Override defaults with user-defined values.
    for k, v in config[name].items():
        kwargs_dict[k] = v
    # Build and return.
    logger = get_logger()
    clz = name_to_clz[name]
    logger.info('building diora name = {}, class = {}'.format(name, clz))
    logger.info('and kwargs = {}'.format(json.dumps(kwargs_dict)))
    return clz.from_kwargs_dict(kwargs_dict)


def get_default_configs(options, context):
    #
    size = 400
    normalize = 'unit'
    n_layers = 2

    res = {'mlp': dict(size=size, outside=True, normalize=normalize, n_layers=n_layers),
           'topk-mlp': dict(size=size, outside=True, normalize=normalize, n_layers=n_layers)}
    return res
