import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import QG


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.clip = args.clip
        self.device = args.device

        self.model = QG(args).to(self.device)
        self.params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=args.lr)

        self.loss_q_rec = 0

    def train(self, c_ids, q_ids, a_ids):
        self.model = self.model.train()

        # Forward
        loss = self.model(c_ids, q_ids, a_ids)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Step
        clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()

        self.loss_q_rec = loss.item()

    def save(self, filename):
        params = {
            'state_dict': self.model.state_dict(),
            'args': self.args
        }
        torch.save(params, filename)

    def reduce_lr(self):
        self.optimizer.param_groups[0]['lr'] *= 0.5
