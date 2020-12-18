import numpy as np
import random
import sys
import utils
import pandas as pd


def train_perceptron(my_data, hidden_size=200):
    error_gr = []
    valer_gr = []
    size = (len(my_data) // 100) * 10
    training_data = my_data.loc[size:, 'id':'fi30']
    validation_data = my_data.loc[:size, 'id':'fi30']
    alpha = 0.0005
    iterations = 150
    np.random.seed(1)

    inputs_size = 7
    num_outs = 2
    hidden_size = 200

    weights1 = 0.2 * np.random.random((inputs_size, hidden_size)) - 0.1
    weights2 = 0.2 * np.random.random((hidden_size, hidden_size)) - 0.1
    weights3 = 0.2 * np.random.random((hidden_size, hidden_size)) - 0.1
    weights4 = 0.2 * np.random.random((hidden_size, num_outs)) - 0.1
    b1 = 0.0
    b2 = 0.0
    b3 = 0.0
    b4 = 0.0

    for j in range(iterations):
        error1 = 0
        val_error1 = 0
        for index, row in training_data.iterrows():
            fit1 = (row['fi7'] - training_data['fi7'].min()) / (training_data['fi7'].max() - training_data['fi7'].min())
            fit2 = (row['fi11'] - training_data['fi11'].min()) / (
                    training_data['fi11'].max() - training_data['fi11'].min())
            fit3 = (row['fi1'] - training_data['fi1'].min()) / (training_data['fi1'].max() - training_data['fi1'].min())
            fit4 = (row['fi28'] - training_data['fi28'].min()) / (
                    training_data['fi28'].max() - training_data['fi28'].min())
            fit5 = (row['fi8'] - training_data['fi8'].min()) / (training_data['fi8'].max() - training_data['fi8'].min())
            fit6 = (row['fi23'] - training_data['fi23'].min()) / (
                    training_data['fi23'].max() - training_data['fi23'].min())
            fit7 = (row['fi24'] - training_data['fi24'].min()) / (
                    training_data['fi24'].max() - training_data['fi24'].min())

            if row['di'] == 'M':
                t = np.array([1, 0])
            else:
                t = np.array([0, 1])

            input = np.array([fit1, fit2, fit3, fit4, fit5, fit6, fit7])
            input = input.reshape(1, -1)

            layer_1 = utils.tanh(np.dot(input, weights1) + b1)
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = utils.tanh(np.dot(layer_1, weights2) + b2)
            dropout_mask = np.random.randint(2, size=layer_2.shape)
            layer_2 *= dropout_mask * 2
            layer_3 = utils.tanh(np.dot(layer_2, weights3) + b3)
            dropout_mask = np.random.randint(2, size=layer_3.shape)
            layer_3 *= dropout_mask * 2
            layer_4 = utils.softmax(np.dot(layer_3, weights4) + b4)

            layer_4_delta = (t - layer_4) / 1 * layer_4.shape[0]
            layer_3_delta = layer_4_delta.dot(weights4.T) * utils.tanh_deriv(layer_3)
            layer_2_delta = layer_3_delta.dot(weights3.T) * utils.tanh_deriv(layer_2)
            layer_1_delta = layer_2_delta.dot(weights2.T) * utils.tanh_deriv(layer_1)

            layer_1_delta *= dropout_mask
            layer_2_delta *= dropout_mask
            layer_3_delta *= dropout_mask

            weights4 += alpha * layer_3.T.dot(layer_4_delta)
            b4 += alpha * layer_4_delta
            weights3 += alpha * layer_2.T.dot(layer_3_delta)
            b3 += alpha * layer_3_delta
            weights2 += alpha * layer_1.T.dot(layer_2_delta)
            b2 += alpha * layer_2_delta

            weights1 += alpha * input.T.dot(layer_1_delta)
            b1 += alpha * layer_1_delta
            error1 += utils.cross_entropy1(layer_4[0], t)

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

            if row['di'] == 'M':
                t = np.array([1, 0])
            else:
                t = np.array([0, 1])

            input_s = np.array([fit1, fit2, fit3, fit4, fit5, fit6, fit7])
            input_s = input_s.reshape(1, -1)
            layer_1 = utils.tanh(input_s.astype(np.float64).dot(weights1) + b1)
            layer_2 = utils.tanh(layer_1.astype(np.float64).dot(weights2) + b2)
            layer_3 = utils.tanh(layer_2.astype(np.float64).dot(weights3) + b3)

            layer_4 = utils.softmax(layer_3.astype(np.float64).dot(weights4) + b4)
            val_error1 += utils.cross_entropy1(layer_4[0], t)

        print(f'Train error {error1 / len(training_data)} ---- Validation error {val_error1 / len(validation_data)} ')
        valer_gr.append(val_error1 / len(validation_data))
        error_gr.append(error1 / len(training_data))
    w1 = weights1
    w2 = weights2
    w3 = weights3
    w4 = weights4

    utils.write_weights(w1, b1, w2, b2, w3, b3, w4, b4)
    return error_gr, valer_gr


def main():
    try:
        training_data= pd.read_csv(sys.argv[1])
        error_y, val_y = train_perceptron(training_data)
        utils.visual_of_learning(error_y, val_y)
    except:
        print('Please add the data')


if __name__ == '__main__':
    main()
