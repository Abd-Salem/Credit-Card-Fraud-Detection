import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.model_selection import train_test_split


class FocalLoss(nn.Module):
    '''
    focal loss class implementation
    '''
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
    '''
    neural network class implementation
    '''
    def __init__(self, input_dim:int, hidden_layers:tuple, dropout:float|int=0.2 ,* ,
                 epochs=100,patience=30,train_val_size=0.2,
                 optimizer:str='adam',optimizer_lr:float|int=0.001, optimizer_weight_decay:float|int=0.0001 ,
                 focal_loss_alpha:float|int=0.25, focal_loss_gamma:int=2):
        super().__init__()      # init parent

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.train_val_size = train_val_size
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.history = {'train_loss':[], 'val_loss':[]}

        # set fl, network struct and optimizer
        self.focal_loss = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)     # init cost fun
        self.net = self._init_network()    # init network
        self.optim = self._init_optimizer()   # init optimizer



    def _init_network(self):
        '''
        initiate network structure
        :return: network
        '''
        # create the network
        in_dim = self.input_dim
        layers = []
        for h in self.hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(self.hidden_layers[-1], 1))
        return nn.Sequential(*layers)

    def _init_optimizer(self):
        '''
        initiate optimizer of training
        :return: None
        '''
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)
        return None

    def forward(self, x):
        return self.net(x).squeeze(-1)


    def _train_step(self, X, t):
        '''
        model training
        :param X: input features
        :param t: target features
        :return: loss scaler value
        '''
        self.train()  # prepare for training mode
        self.optim.zero_grad()  # clear any past saves
        logits = self(X)  # get logits
        loss = self.focal_loss(logits, t)  # get loss object
        loss.backward()  # back propagation
        self.optim.step()  # update weights
        return loss.item()

    def predict_proba(self, X):
        '''
        prediction method
        :param X: input features
        :return: predicted probabilities
        '''
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        self.eval()  # prepare for inference mode
        with (torch.no_grad()):
            proba = torch.sigmoid(self(X)).numpy()
            return np.column_stack([1-proba, proba])


    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, X, t):
        '''
        evaluation method
        :param X: input features
        :param t: target features
        :return:
        '''
        self.eval()  # prepare for inference
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        with torch.no_grad():
            logits = self(X)
            loss = self.focal_loss(logits, t)  # get loss
        return loss.item()

    def fit(self,X,t):
        '''
        fit input features to target
        :param X: input features
        :param t: target
        :return: None
        '''

       # split data and normalize dtype
        x_tr, x_val, t_tr, t_val = train_test_split(X,t,test_size=self.train_val_size)
        x_tr = torch.tensor(x_tr, dtype=torch.float32)
        t_tr = torch.tensor(t_tr, dtype=torch.float32)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        t_val = torch.tensor(t_val, dtype=torch.float32)

        self.history['train_loss'] ,self.history['val_loss'] = [], []

        # training process (epochs with early stop)
        best_val_score = float('inf')
        patience_counter = 0
        best_weights = None
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_step(X=x_tr, t=t_tr)
            val_loss = self.evaluate(X=x_val, t=t_val)

            # update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if best_val_score > val_loss:
                best_val_score = val_loss  # update best val
                patience_counter = 0  # make patience counter to zero
                best_weights = self.state_dict()  # get best weights
            else:
                patience_counter += 1  # no updates in val score
                if patience_counter >= self.patience:
                    break  # no convergence

        if not best_weights is None:
            self.load_state_dict(best_weights)  # update with optimal weights

    def get_params(self, deep=True):
         return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'patience': self.patience,
            'train_val_size': self.train_val_size,
            'optimizer': self.optimizer,
            'optimizer_lr': self.optimizer_lr,
            'optimizer_weight_decay': self.optimizer_weight_decay,
            'focal_loss_alpha': self.focal_loss_alpha,
            'focal_loss_gamma': self.focal_loss_gamma
        }.copy()

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

        # set each params method
        net_params = {'input_dim', 'hidden_layers', 'dropout'}
        optim_params = {'optimizer','optimizer_lr', 'optimizer_weight_decay'}
        fl_params = {'focal_loss_alpha', 'focal_loss_gamma'}

        # check params matching to call methods
        if params.keys() & net_params:
            self.net = self._init_network()
        if params.keys() & optim_params:
            self.optim = self._init_optimizer()
        if params.keys() & fl_params:
            self.focal_loss = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)

        return self

