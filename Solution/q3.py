import numpy as np


ita = 1.0

def forward(X, W, b):
	return W.dot(X) + b


def loss(logits, y):
    logitTrans = logits - np.max(logits, axis=0)
    expLogit = np.exp(logitTrans)
    logp = logitTrans - np.log(np.sum(expLogit, axis=0))
    numInstance = y.shape[1]
    return - np.sum(y * logp) / numInstance


def dOut(logits, y):
    logitTrans = logits - np.max(logits, axis=0)
    expLogit = np.exp(logitTrans)
    logp = logitTrans - np.log(np.sum(expLogit, axis=0))
    numInstance = y.shape[1]

    p = np.exp(logp)
    return p, (p - y) / numInstance


def backward(W, b, dZ, X):
    dW = dZ.dot(X.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    W -= ita * dW
    b -= ita * db
    return W, b, dW, db


def main():
    W = np.zeros((2, 2))
    b = np.zeros((2, 1))

    X = np.eye(2)
    Y = np.eye(2)

    i = 0
    while True:
        i += 1
        Z = forward(X, W, b)
        softm = np.exp(Z) / np.sum(np.exp(Z), axis = 0)
        print('out:', softm)
        p, dZ = dOut(Z, Y)
        l = loss(Z, Y)
        W, b, dW, db = backward(W, b, dZ, X)
        print('After %d iteration:'%i)
        print('z :\n', Z)
        print('dz:\n', dZ)
        print('p :\n', p)
        print('dW:\n', dW)
        print('db:\n', db)
        print('W :\n', W)
        print('b :\n', b)
        print('loss:\n', l)

        a = input()

main()