import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
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

    def forward(self, inputs, targets):
        targets = targets.float()
        inputs = inputs.float()
        bce = f.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        if torch.isnan(bce).any():
            raise ValueError("NaN in bce of focal loss")

        pt = torch.exp(-bce)
        # pt = torch.clamp(pt, 1e-8, 1.0 - 1e-8)
        alpha_t = self.alpha * targets + (1-self.alpha) * (1-targets)

        if torch.isnan(bce).any():
            raise ValueError("NaN in alpha_t of focal loss")


        loss = alpha_t * (1-pt)**self.gamma * bce

        if torch.isnan(loss).any():
            raise ValueError("NaN loss in focal loss")

        return loss.mean()



class _FraudDataSet(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



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
        build network structure
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
                 dropout:float=0.2, epochs:int=100, patience:int=50, train_val_size:float=0.1, batch_size=32,
                 optimizer_lr:float|int=0.001, optimizer_weight_decay:float|int=0.0001 ,
                 focal_loss_alpha:float|int=0.25, focal_loss_gamma:int=2,
                 threshold=0.5, random_state=42):
        '''
        :param input_dim: num of input features
        :param hidden_layers: structure of hidden layers
        :param activation: activation function
        :param dropout: dropout value
        :param epochs: num of epochs
        :param patience: early stop counter
        :param train_val_size: ratio of splitting
        :param batch_size: batch size
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
        self.batch_size = batch_size
        self.random_state = random_state
        self.opt_lr = optimizer_lr
        self.opt_wd = optimizer_weight_decay
        self.fl_alpha = focal_loss_alpha
        self.fl_gamma = focal_loss_gamma
        self.th = threshold
        self.history = {'train_loss':[], 'val_loss':[]}


        self.net_ = None
        self.optimizer_ = None
        self.focal_loss_ = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        logits = self.net_(X)  # forward
        if torch.isnan(logits).any():
            raise ValueError("NaN in train logits")

        loss = self.focal_loss_(logits, t)  # cost

        self.optimizer_.zero_grad()  # clear
        loss.backward()  # back propagation
        # torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
        self.optimizer_.step()  # update weights
        return loss.item()


    def evaluate(self, X, t):
        '''
        evaluation method
        :param X: input features
        :param t: target features
        :return:
        '''
        # if not isinstance(X, torch.Tensor):
        #     X = torch.tensor(np.array(X), dtype=torch.float32)
        # if not isinstance(t, torch.Tensor):
        #     t = torch.tensor(np.array(t), dtype=torch.float32)

        self.net_.eval()  # prepare for inference
        with torch.no_grad():
            logits = self.net_(X)   # forward

            if torch.isnan(logits).any():
                raise ValueError("NaN in eval logits")

            loss = self.focal_loss_(logits, t)  # cost
        return loss.item()


    def fit(self, X,t):
        '''
        fit input features to target
        :param X: input features
        :param t: target
        :return: None
        '''

        if self.input_dim is None:
            self.input_dim = X.shape[1]

        torch.manual_seed(self.random_state)    # reproducibility

        # init model, cost and optimizer
        self.net_ = _MLPNet(self.input_dim, self.hidden_layers, self.activation,self.dropout)
        self.focal_loss_ = _FocalLoss(alpha=self.fl_alpha, gamma=self.fl_gamma)
        self.optimizer_ = torch.optim.AdamW(self.net_.parameters(), lr=self.opt_lr, weight_decay=self.opt_wd)

       # split data and normalize dtype
        x_tr, x_val, t_tr, t_val = train_test_split(X,t,test_size=self.tv_ratio,
                                                    stratify=t, random_state=self.random_state)

        # Convert and move to device
        x_tr = torch.tensor(x_tr, dtype=torch.float32, device=self.device)
        t_tr = torch.tensor(t_tr.values, dtype=torch.float32, device=self.device)
        x_val = torch.tensor(x_val, dtype=torch.float32, device=self.device)
        t_val = torch.tensor(t_val.values, dtype=torch.float32, device=self.device)

        # history
        self.history['train_loss'] ,self.history['val_loss'] = [], []

        # training process (epochs with early stop)
        best_val_score = float('inf')
        patience_counter = 0
        best_weights = None

        # batching data
        dataset = _FraudDataSet(x_tr, t_tr)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0      # start value
            for X_batch, y_batch in data_loader:
                epoch_loss += self._train_step(X=X_batch, t=y_batch)    # add loss

            avg_loss = epoch_loss / len(data_loader)    # calculate avg
            val_loss = self.evaluate(X=x_val, t=t_val)  # validation loss

            # update history
            self.history['train_loss'].append(avg_loss)
            self.history['val_loss'].append(val_loss)

            # check for best loss
            if val_loss < best_val_score:
                best_val_score = val_loss
                patience_counter = 0
                best_weights = copy.deepcopy(self.net_.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if not best_weights is None:
            self.net_.load_state_dict(best_weights)  # update with optimal weights

        return self



    def predict_proba(self, X):
        '''
        prediction method
        :param X: input features
        :return: predicted probabilities
        '''

        if self.net_ is None:
            raise Exception('Model isn\'t fitted yet!')

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.array(X), dtype=torch.float32)

        X = X.to(self.device)

        self.net_.eval()  # prepare for inference mode
        with (torch.no_grad()):
            logits = self.net_(X)

            if torch.isnan(logits).any():
                raise ValueError("NaN in prediction logits")

            probs = torch.sigmoid(logits).numpy()

            if np.isnan(probs).any() or np.isinf(probs).any():
                raise ValueError("Invalid probabilities (NaN or Inf)")

            return np.column_stack([1-probs, probs])


    def predict(self, X):
        '''
        return prediction according threshold (default=0.5)
        :param X: input features
        :return: predicted classes
        '''
        return (self.predict_proba(X)[:, 1] >= self.th).astype(int)


    def score(self, X, y):
        probs = self.predict_proba(X)[:, 1]

        if np.isnan(probs).any():
            raise ValueError("NaN detected before scoring")

        return average_precision_score(y, probs)


    def get_params(self):
         return {
            'hidden_layers': self.hidden_layers,
            'activation' : self.activation,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'optimizer': 'adamw',
            'optimizer_lr': self.opt_lr,
            'optimizer_weight_decay': self.opt_wd,
            'focal_loss_alpha': self.fl_alpha,
            'focal_loss_gamma': self.fl_gamma
        }.copy()



class MLP_Pipeline():
    def __init__(self, steps):
        self.transformers = [transformer for _, transformer in steps[:-1]]

        self.model = steps[-1][1]


    def _prepare_data(self, X):
        for transformer in self.transformers:
            X = transformer.transform(X)
        return X

    def fit(self, X, t):
        for transformer in self.transformers:
            X = transformer.fit_transform(X)
        self.model.fit(X, t)


    def predict_proba(self, X):
        X = self._prepare_data(X)
        return self.model.predict_proba(X)

    def predict(self, X):
        X = self._prepare_data(X)
        return self.model.predict(X)

    def score(self, X, t):
        X = self._prepare_data(X)
        return self.model.score(X, t)

    def get_params(self):
        return self.model.get_params()