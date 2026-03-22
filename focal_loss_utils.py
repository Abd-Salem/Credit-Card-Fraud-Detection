import torch
import numpy as np

def train(model, X, t, criterion, optimizer):
    '''
    model training
    :param model: torch.nn model
    :param X: input features
    :param t: target features
    :param criterion: cost function
    :param optimizer: type of optimizing technique (adam, sgd, etc.)
    :return: loss scaler value
    '''

    model.train()   # prepare for training mode
    optimizer.zero_grad()   # clear any past saves
    logits = model(X)     # get logits
    loss = criterion(logits, t)   # get loss object
    loss.backward()     # back propagation
    optimizer.step()    # update weights
    return loss.item()


def predict(model, X):
    '''
    prediction method
    :param model: torch.nn model
    :param X: input features
    :return: predicted probabilities
    '''
    model.eval()    # prepare for inference mode
    with torch.no_grad():
        logits = model(X)
        proba = torch.sigmoid(logits)   # get probabilities
    return np.array(proba)


def evaluate(model, X, t, criterion):
    '''
    evaluation method
    :param model: torch.nn model
    :param X: input features
    :param t: target features
    :param criterion: cost function
    :return:
    '''
    model.eval()    # prepare for inference
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, t)    # get loss
    return loss.item()
