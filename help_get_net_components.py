import torch
import torch.nn as nn
from help_get_diora import get_diora
from help_get_loss_funcs import get_loss_funcs


def get_net_components(options, context):
    #
    embeddings = context['embeddings']
    word2idx = context['word2idx']
    batch_iterator = context['batch_iterator']

    # TODO: There should be a better way to do this?
    options.input_dim = embeddings.shape[1]
    if options.projection == 'word2vec':
        options.input_dim = embeddings.shape[1]
    elif options.projection == 'elmo':
        options.input_dim = 1024
    elif options.projection == 'bert':
        options.input_dim = 768
    elif options.projection == 'mask':
        raise NotImplementedError

    # Embed
    projection_layer, embedding_layer = get_embed_and_project(options, embeddings, options.input_dim, options.hidden_dim, word2idx)

    # Diora
    diora_context = {}
    diora_context['projection_layer'] = projection_layer
    diora = get_diora(options, diora_context, config=options.model_config)

    # Loss
    loss_context = {}
    loss_context['batch_iterator'] = batch_iterator
    loss_context['embedding_layer'] = embedding_layer
    loss_context['diora'] = diora
    loss_context['word2idx'] = word2idx
    loss_context['embeddings'] = embeddings
    loss_context['cuda'] = options.cuda
    loss_context['input_dim'] = options.input_dim
    loss_context['projection_layer'] = projection_layer
    loss_funcs = get_loss_funcs(options, loss_context, config_lst=options.loss_config)

    # Return components.
    components = {}
    components['projection_layer'] = projection_layer
    components['embedding_layer'] = embedding_layer
    components['diora'] = diora
    components['loss_funcs'] = loss_funcs
    return components


def get_embed_and_project(options, embeddings, input_dim, size, word2idx=None, contextual=False):
    if options.projection == 'word2vec':
        projection_layer, embedding_layer = word2vec_projection(options, embeddings, input_dim, size, word2idx)
    else:
        raise NotImplementedError(f'unknown projection conf {options.projection}')
    return projection_layer, embedding_layer


def word2vec_projection(options, embeddings, input_dim, size, word2idx=None):
    embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
    elmo = None
    projection_layer = EmbedAndProject(embedding_layer, input_size=input_dim, size=size)
    return projection_layer, embedding_layer


class EmbedAndProject(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def embed(self, x):
        """ Always context-insensitive embedding.
        """
        return self.embeddings(x)

    def project(self, x):
        return torch.matmul(x, self.mat.t())

    def forward(self, x, info=None):
        batch_size, length = x.shape

        embed = self.embed(x)

        out = embed

        return self.project(out)

