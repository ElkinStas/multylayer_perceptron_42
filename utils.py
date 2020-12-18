import numpy as np
import matplotlib.pyplot as plt
import sys

def tanh(x):
    return np.tanh(x)


def tanh_deriv(output):
    return 1 - (output ** 2)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def cross_entropy1(predictions, targets, epsilon=1e-12):

    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def visual_of_learning(error_y, val_y):
    x = []
    iterator = 0
    for i in range(len(error_y)):
        x.append((iterator))
        iterator += 1

    y1 = error_y
    y2 = val_y
    plt.figure(figsize=(12, 7))
    plt.plot(x, y1, '-', alpha=0.7, label="training", lw=2, mec='b', mew=2, ms=10)
    plt.plot(x, y2, '-', label="validation", mec='r', lw=2, mew=2, ms=10)
    plt.legend()
    plt.grid(True)

    plt.title('Learning curve')
    plt.show()



def write_weights(w1,b1,w2,b2,w3,b3,w4,b4):
    np.savetxt("w1.csv", w1, delimiter=",")
    np.savetxt("b1.csv", b1, delimiter=",")
    np.savetxt("w2.csv", w2, delimiter=",")
    np.savetxt("b2.csv", b2, delimiter=",")
    np.savetxt("w3.csv", w3, delimiter=",")
    np.savetxt("b3.csv", b3, delimiter=",")
    np.savetxt("w4.csv", w4, delimiter=",")
    np.savetxt("b4.csv", b4, delimiter=",")

class Perceptron:
    def __init__(self, n_inp, n_out):
        self.shape = (n_inp, n_out)

    def __call__(self, x, w1, b1, w2, b2, w3, b3, w4, b4):
        self.activations1 = tanh(x.astype(np.float64).dot(w1) + b1)
        self.activations2 = tanh(self.activations1.astype(np.float64).dot(w2) + b2)
        self.activations3 = tanh(self.activations2.astype(np.float64).dot(w3) + b3)
        self.activations4 = softmax(self.activations3.astype(np.float64).dot(w4) + b4)
        return self.activations4
