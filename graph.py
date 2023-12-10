import numpy as np
import matplotlib.pyplot as plt
from base_model_utils import import_scaled_data


def graph_stuff(seedling):
    # import reading data
    data = import_scaled_data()

    # import prediction data
    prediction_data = np.load(f'output/data_{seedling}.npz')
    preds = prediction_data['preds']
    weights = prediction_data['weights']
    bm_1 = prediction_data['bm_1']
    bm_2 = prediction_data['bm_2']

    # fix length
    length = min([len(l) for l in [data, preds, weights, bm_1, bm_2]])
    length = length - 2993
    data = data[-length:]
    preds = preds[-length:]
    weights = weights[-length:]
    bm_2 = bm_2[-length:]
    bm_1 = bm_1[-length:]

    # plot
    plt.plot(data, label="data", color='black')
    plt.plot(bm_1, label="bm_1", color='blue')
    plt.plot(bm_2, label="bm_2", color='green')
    plt.plot(preds, label="preds", color='red')
    plt.plot(weights[:, 0], label="weights", color='gray')

    # labeling
    plt.xlabel('15 minute time steps')
    plt.ylabel('Degrees F')
    plt.title('Synthetic Model Predictions')
    plt.legend()

    # Display the plot
    plt.show()



if __name__ == "__main__":
    graph_stuff(3)