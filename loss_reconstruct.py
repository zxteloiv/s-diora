import torch
import torch.nn as nn


def scores_for_tokens(tokens, cell, embeddings, mat):
    assert len(tokens.shape) == 1
    # cell: (10, 18, 400)
    batch_size, length, size = cell.shape
    # emb: (1, 109, 1024)
    # embeddings: (124, 1024)
    emb = embeddings(tokens.unsqueeze(0))
    # cell: (10, 18, 1, 400)
    cell = cell.view(batch_size, length, 1, size)
    # mat: (400, 1024)
    # proj: (1, 109, 400)
    proj = torch.matmul(emb, torch.t(mat))
    # score: (10, 18, 109)
    score = torch.einsum('zec,abxc->abe', proj, cell)
    return score


class ReconstructFixedVocab(nn.Module):
    name = 'reconstruct_softmax_v2_loss'

    def __init__(self, embeddings, input_size, size, margin=1, path='./resource/ptb_top_10k.txt',
                 word2idx=None, cuda=False, load=True, skip=False):
        super().__init__()
        self.load = load
        self.path = path
        self.margin = margin
        self.input_size = input_size
        self.skip = skip

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

        # read vocab
        vocab = []
        with open(path) as f:
            for line in f:
                # TODO: Support OOV.
                w, count = line.strip().split()
                if w not in word2idx:
                    continue
                vocab.append(word2idx[w])

        self.vocab = vocab

    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        kwargs_dict['embeddings'] = context['embedding_layer']
        kwargs_dict['cuda'] = context['cuda']
        kwargs_dict['word2idx'] = context['word2idx']
        return cls(**kwargs_dict)

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def reconstruct(self, sentences, cell):
        batch_size, length = sentences.shape
        device = torch.cuda.current_device() if self._cuda else None

        vocab = self.vocab
        embeddings = self.embeddings
        mat = self.mat

        # output_tokens_tensor: (109,)
        output_tokens_tensor = torch.tensor(vocab, dtype=torch.long, device=device)
        # cell: (10, 18, 400)
        # embeddings: (124, 1024)
        # mat: (400, 1024)
        # scores: (10, 18, 109)
        scores = scores_for_tokens(output_tokens_tensor, cell, embeddings, mat)

        # Vocab index.
        found = 0   # int, 155
        vocab_index = []
        for x in sentences.view(-1).tolist():
            if x in vocab:
                vocab_index.append(vocab.index(x))
                found += 1
            else:
                vocab_index.append(-1)
        vocab_index = torch.tensor(vocab_index, dtype=torch.long, device=device)    # (180,)
        mask = vocab_index >= 0
        loss = nn.CrossEntropyLoss()(scores.view(batch_size * length, -1)[mask], vocab_index[mask])

        return loss
