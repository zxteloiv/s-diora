import os

from eval_parsing import ParsingComponent
from evaluation_utils import BaseEvalFunc
from experiment_logger import get_logger
from help_get_loss_funcs import get_component_factory

name_to_clz = {
    'unlabeled_binary_parsing': ParsingComponent,
}


def build_eval_components(name, kwargs_dict, context):
    clz = name_to_clz[name]
    logger = get_logger()
    logger.info('building eval component name = {}, class = {}'.format(name, clz))
    logger.info('with kwargs = {}'.format(kwargs_dict))
    return clz.from_kwargs_dict(context, kwargs_dict)


class ModelEvaluation(object):
    def __init__(self, components):
        super(ModelEvaluation, self).__init__()
        self.validate(components)
        self.components = components

    def validate(self, components):
        check = set([func.name for func in components])
        assert len(check) == len(components), "Each name must be unique."

    def run(self, trainer, info, metadata):
        for func in self.components:
            assert isinstance(func, BaseEvalFunc), "All eval funcs should be subclass of BaseEvalFunc."
            assert hasattr(func, 'is_initialized') and func.is_initialized, \
                "Do not override __init__ for BaseEvalFunc, instead use `init_defaults` or `from_kwargs_dict`."
            if not func.enabled:
                continue
            outfile = os.path.join(info['experiment_path'], 'eval_{}.step_{}.txt'.format(func.name, info['step']))
            info['outfile'] = outfile
            result = func.run(trainer, info)
            yield {'result': result, 'component': func}


def get_default_configs(options):
    res = {}
    res['unlabeled_binary_parsing'] = dict()
    return res


get_eval_components = get_component_factory(build_eval_components, get_default_configs)
