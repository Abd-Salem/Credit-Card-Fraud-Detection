import torch
import torch.nn as nn
import torch.nn.functional as f
from configs import Config
import numpy as np

config = Config()



class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,inputs, targets):
        bce = f.binary_cross_entropy_with_logits(inputs,targets,reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1-self.alpha) * (1-targets)
        loss = alpha_t * (1-pt)**self.gamma * bce
        return loss.mean()

class MLP_FL(nn.Module):
    def __init__(self, input_dim, hidden_layers=(128,64), dropout=0.2):
        super().__init__()
        in_dim = input_dim
        layers = []
        for h in layers:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(model, x_train, t_train, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(x_train)
    loss = criterion(logits, t_train)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, x_train):
    model.eval()
    with torch.no_grad():
        logits = model(x_train)
        proba = torch.sigmoid(logits)
    return np.array(proba)


def evaluate(model, x_train, t_train, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x_train)
        loss = criterion(logits, t_train)
    return loss.item()
