import json

from experiment_logger import get_logger
from loss_reconstruct import ReconstructFixedVocab
from loss_greedy_reconstruct import GreedyReconstruct


name_to_clz = {
    'reconstruct_softmax_v2': ReconstructFixedVocab,
    'greedy_reconstruct_loss': GreedyReconstruct,
}


def build_loss(name, kwargs_dict, context):
    clz = name_to_clz[name]
    logger = get_logger()
    logger.info('building loss component name = {}, class = {}'.format(name, clz))
    logger.info('with kwargs = {}'.format(kwargs_dict))
    return clz.from_kwargs_dict(context, kwargs_dict)


def get_default_configs(options):
    mlp_input_dim = options.hidden_dim

    return {
        'reconstruct_softmax_v2': dict(input_size=options.input_dim, size=mlp_input_dim),
        'greedy_reconstruct_loss': dict(input_size=options.input_dim, size=mlp_input_dim),
    }


def get_component_factory(builder_func, get_default_conf):
    def fac(options, context, config_lst=None):
        assert isinstance(config_lst, (list, tuple)), "There should be a `list` of configs."

        result = []

        for i, config_str in enumerate(config_lst):
            config = json.loads(config_str)

            assert isinstance(config, dict), "Config[{}] with value {} is not type dict.".format(i, config)

            assert len(config.keys()) == 1, "Each config should have 1 key only."

            name = list(config.keys())[0]

            if not config[name].get('enabled', True):
                continue

            # Use default config if it exists.
            kwargs_dict = get_default_conf(options).get(name, {})
            # Override defaults with user-defined values.
            for k, v in config[name].items():
                kwargs_dict[k] = v
            # Build and append component.
            result.append(builder_func(name, kwargs_dict, context))

        return result

    return fac


get_loss_funcs = get_component_factory(build_loss, get_default_configs)

