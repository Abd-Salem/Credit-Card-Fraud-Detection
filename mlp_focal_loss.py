import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


class _FocalLoss(nn.Module):
    '''
    focal loss class implementation
    '''
    def __init__(self,alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,inputs, targets):
        bce = f.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1-self.alpha) * (1-targets)
        loss = alpha_t * (1-pt)**self.gamma * bce
        return loss.mean()


class _MLPNet(nn.Module):
    '''
    neural network class implementation
    '''
    def __init__(self, input_dim:int, hidden_layers:tuple=(100,), activation='relu' , dropout:float=0.2):
        super().__init__()      # init parent

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout

        # set fl, network struct and optimizer
        self.net = self._build()    # init network


    def _build(self):
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


    def forward(self, x):
        return self.net(x).squeeze(-1)



class MLP_FL:
    '''
    neural network with focal loss
    '''


    def __init__(self, input_dim:int=None, hidden_layers:tuple=(100,), activation='relu' , *,
                 dropout:float=0.2, epochs:int=100, patience:int=50, train_val_size:float=0.1,
                 optimizer:str='adam', optimizer_lr:float|int=0.001, optimizer_weight_decay:float|int=0.0001 ,
                 focal_loss_alpha:float|int=0.25, focal_loss_gamma:int=2,threshold=0.5, scaler=None ,random_state=42):
        '''
        :param input_dim: num of input features
        :param hidden_layers: structure of hidden layers
        :param activation: activation function
        :param dropout: dropout value
        :param epochs: num of epochs
        :param patience: early stop counter
        :param train_val_size: ratio of splitting
        :param optimizer: name of optimizer
        :param optimizer_lr: learning rate
        :param optimizer_weight_decay: weight decay
        :param focal_loss_alpha: alpha
        :param focal_loss_gamma: gamma
        :param threshold: prediction threshold
        :param random_state: random state
        '''

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.tv_ratio = train_val_size
        self.random_state = random_state
        self.opt_type = optimizer
        self.opt_lr = optimizer_lr
        self.opt_wd = optimizer_weight_decay
        self.fl_alpha = focal_loss_alpha
        self.fl_gamma = focal_loss_gamma
        self.th = threshold
        self.history = {'train_loss':[], 'val_loss':[]}


        self.net_ = None
        self.optimizer_ = None
        self.focal_loss_ = None
        self.scaler = scaler



    def _train_step(self, X, t):
        '''
        model training
        :param X: input features
        :param t: target features
        :return: loss scaler value
        '''
        if self.net_ is None:
            raise Exception('No Fitted data !!')

        self.net_.train()  # prepare for training mode
        self.optimizer_.zero_grad()  # clear any past saves
        logits = self.net_(X)  # get logits
        loss = self.focal_loss_(logits, t)  # get loss object
        loss.backward()  # back propagation
        self.optimizer_.step()  # update weights
        return loss.item()


    def evaluate(self, X, t):
        '''
        evaluation method
        :param X: input features
        :param t: target features
        :return:
        '''
        self.net_.eval()  # prepare for inference
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(np.array(t), dtype=torch.float32)
        with torch.no_grad():
            logits = self.net_(X)
            loss = self.focal_loss_(logits, t)  # get loss
        return loss.item()


    def fit(self,X,t):
        '''
        fit input features to target
        :param X: input features
        :param t: target
        :return: None
        '''

        if self.input_dim is None:
            self.input_dim = X.shape[1]
        self.net_ = _MLPNet(self.input_dim, self.hidden_layers, self.activation,self.dropout)
        self.focal_loss_ = _FocalLoss(alpha=self.fl_alpha, gamma=self.fl_gamma)
        self.optimizer_ = torch.optim.Adam(self.net_.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd)

       # split data and normalize dtype
        x_tr, x_val, t_tr, t_val = train_test_split(X,t,test_size=self.tv_ratio,
                                                    stratify=t, random_state=self.random_state)

        # if scaler object is provided
        if self.scaler is not None and hasattr(self.scaler, 'fit_transform'):
            x_tr = self.scaler.fit_transform(x_tr)
            x_val = self.scaler.transform(x_val)

        x_tr = torch.tensor(np.array(x_tr), dtype=torch.float32)
        t_tr = torch.tensor(np.array(t_tr), dtype=torch.float32)
        x_val = torch.tensor(np.array(x_val), dtype=torch.float32)
        t_val = torch.tensor(np.array(t_val), dtype=torch.float32)

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
                best_weights = self.net_.state_dict()  # get best weights
            else:
                patience_counter += 1  # no updates in val score
                if patience_counter >= self.patience:
                    break  # no convergence

        if not best_weights is None:
            self.net_.load_state_dict(best_weights)  # update with optimal weights

        return self



    def predict_proba(self, X):
        '''
        prediction method
        :param X: input features
        :return: predicted probabilities
        '''

        # For validation data (no change if X is train data)
        if self.scaler is not None and hasattr(self.scaler, 'transform'):
            X = self.scaler.transform(X)

        if self.net_ is None:
            raise Exception('Model not fitted yet!')

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)

        self.net_.eval()  # prepare for inference mode
        with (torch.no_grad()):
            proba = torch.sigmoid(self.net_(X)).cpu().numpy()
            return np.column_stack([1-proba, proba])


    def predict(self, X):
        '''
        return prediction according threshold (default=0.5)
        :param X: input features
        :return: predicted classes
        '''
        return (self.predict_proba(X)[:, 1] >= self.th).astype(int)

    def score(self, X, y):
        return average_precision_score(y, self.predict_proba(X)[:, 1])


    def get_params(self):
         return {
            'scaler' : self.scaler,
            'hidden_layers': self.hidden_layers,
            'activation' : self.activation,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'optimizer': self.opt_type,
            'optimizer_lr': self.opt_lr,
            'optimizer_weight_decay': self.opt_wd,
            'focal_loss_alpha': self.fl_alpha,
            'focal_loss_gamma': self.fl_gamma
        }.copy()