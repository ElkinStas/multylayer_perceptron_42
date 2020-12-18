import numpy as np
import sys
import utils
import pandas as pd
from numpy import genfromtxt


def predictor(my_data, w1, b1, w2, b2, w3, b3, w4, b4):
    validation_data = my_data
    val_error1 = 0
    network = utils.Perceptron(7, 2)
    for index, row in validation_data.iterrows():
        fit1 = (row['fi7'] - validation_data['fi7'].min()) / (
                validation_data['fi7'].max() - validation_data['fi7'].min())
        fit2 = (row['fi11'] - validation_data['fi11'].min()) / (
                validation_data['fi11'].max() - validation_data['fi11'].min())
        fit3 = (row['fi1'] - validation_data['fi1'].min()) / (
                validation_data['fi1'].max() - validation_data['fi1'].min())
        fit4 = (row['fi28'] - validation_data['fi28'].min()) / (
                validation_data['fi28'].max() - validation_data['fi28'].min())
        fit5 = (row['fi8'] - validation_data['fi8'].min()) / (
                validation_data['fi8'].max() - validation_data['fi8'].min())
        fit6 = (row['fi23'] - validation_data['fi23'].min()) / (
                validation_data['fi23'].max() - validation_data['fi23'].min())
        fit7 = (row['fi24'] - validation_data['fi24'].min()) / (
                validation_data['fi24'].max() - validation_data['fi24'].min())
        fit8 = (row['fi14'] - validation_data['fi14'].min()) / (
                validation_data['fi14'].max() - validation_data['fi14'].min())

        if row['di'] == 'M':
            t = np.array([1, 0])
        else:
            t = np.array([0, 1])

        input_s = np.array([fit1, fit2, fit3, fit4, fit5, fit6, fit7])
        input_s = input_s.reshape(1, -1)
        prediction = network(input_s, w1, b1, w2, b2, w3, b3, w4, b4)
        val_error1 += utils.cross_entropy1(prediction[0], t)

    print(f'Validation cross entropy {val_error1 / len(validation_data)} ')


def main():
    try:
        prediction_data = pd.read_csv(sys.argv[1])
        w1 = genfromtxt('w1.csv', delimiter=',')
        b1 = genfromtxt('b1.csv', delimiter=',')
        w2 = genfromtxt('w2.csv', delimiter=',')
        b2 = genfromtxt('b2.csv', delimiter=',')
        w3 = genfromtxt('w3.csv', delimiter=',')
        b3 = genfromtxt('b3.csv', delimiter=',')
        w4 = genfromtxt('w4.csv', delimiter=',')
        b4 = genfromtxt('b4.csv', delimiter=',')
        predictor(prediction_data, w1, b1, w2, b2, w3, b3, w4, b4)
    except:
        print('Please load data to predict')

if __name__ == '__main__':
    main()
