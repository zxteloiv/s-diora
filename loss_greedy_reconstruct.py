import torch
import torch.nn as nn


class SupervisedParsingLoss(nn.Module):
    def __init__(self, margin, print_match=False):
        super().__init__()
        self.margin = margin
        self.print_match = print_match

    def forward(self, gold, pred, gold_trees=None, pred_trees=None):
        batch_size, n_trees = gold.shape
        device, shape = gold.device, gold.shape
        hinge = torch.clamp(pred + self.margin - gold, min=0)

        if gold_trees is not None:
            zeros = torch.zeros(shape, device=device, dtype=torch.float)
            pred_eq_gold = torch.zeros(shape, device=device, dtype=torch.bool)

            # assert
            assert len(gold_trees) == batch_size
            assert len(gold_trees) == len(pred_trees)
            for i_b in range(batch_size):
                assert len(pred_trees[i_b]) == n_trees

            # TODO: Add structure sensitive margin.

            for i_b in range(batch_size):
                hash_gold = tuple(sorted(gold_trees[i_b]))
                for pos in range(n_trees):
                    hash_pred = tuple(sorted(pred_trees[i_b][pos]))
                    pred_eq_gold[i_b, pos] = hash_gold == hash_pred

            if self.print_match:
                print(n_trees, pred_eq_gold.sum(dim=1))

            loss = torch.where(pred_eq_gold, zeros, hinge)
        else:
            loss = hinge

        return loss


class GreedyReconstruct(nn.Module):
    name = 'greedy_reconstruct_loss'

    def __init__(self, embeddings, input_size, size, weight=1, cuda=False,
                 reconstruct_loss=None, train_tree=True, margin=1, print_match=False):
        super().__init__()
        self.reconstruct_loss = reconstruct_loss
        self.margin = margin
        self.weight = weight
        self._cuda = cuda

        self.print_match = print_match
        self.train_tree = train_tree

        self.reset_parameters()

    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        kwargs_dict['embeddings'] = context['embedding_layer']
        kwargs_dict['cuda'] = context['cuda']
        return cls(**kwargs_dict)

    def reset_parameters(self):
        pass

    def reconstruct(self, sentences, cell, info):
        loss_func = getattr(info['net'], self.reconstruct_loss)
        return loss_func.reconstruct(sentences, cell)

    def forward_helper(self, sentences, diora, info):
        batch_size, length = sentences.shape

        # Maximize the reconstruction (wp_loss).
        cell0 = diora.chart['outside_h'][:, :length]
        xent0 = self.reconstruct(sentences, cell0, info)
        wp_loss = xent0.mean()
        loss = wp_loss

        # Maximize the top tree (tr_loss).
        if self.train_tree:
            tr_score_0 = diora.chart['inside_s'][:, -1].contiguous().view(batch_size, 1)
            tr_score_1 = diora.charts[1]['inside_s'][:, -1].contiguous().view(batch_size, 1)
            tr_loss_func = SupervisedParsingLoss(self.margin, print_match=self.print_match)
            tr_loss = tr_loss_func(gold=tr_score_0, pred=tr_score_1).mean()
            loss = loss + tr_loss

        return loss

    def forward(self, *args, **kwargs):
        loss = self.forward_helper(*args, **kwargs)
        ret = {self.name: self.weight * loss}
        return loss, ret
