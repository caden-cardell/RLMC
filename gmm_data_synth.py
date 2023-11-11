import torch
import numpy as np


def synthesize_deltas():
    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Define the means and standard deviations
    dist_1_mean = 0
    dist_1_std_dev = 1

    dist_2_mean = 1
    dist_2_std_dev = 4

    # Define the size of the output tensor
    size = (500,)

    # Generate random deltas
    dist_1_deltas = torch.normal(dist_1_mean, dist_1_std_dev, size)
    dist_1_prediction_deltas = torch.normal(dist_1_mean, dist_1_std_dev, size)

    dist_2_deltas = torch.normal(dist_2_mean, dist_2_std_dev, size)
    dist_2_prediction_deltas = torch.normal(dist_2_mean, dist_2_std_dev, size)

    np.savez('dataset/synthesized_deltas.npz',
             dist_1_deltas=dist_1_deltas,
             dist_1_prediction_deltas=dist_1_prediction_deltas,
             dist_2_deltas=dist_2_deltas,
             dist_2_prediction_deltas=dist_2_prediction_deltas)


def synthesize_static_alpha_data():

    # parameters
    alpha = 0.25
    x_window_size = 50  # the number of elements to use in each prediction
    y_window_size = 5 # the number of elements to use in each prediction

    # import data
    input_data = np.load('dataset/synthesized_deltas.npz')

    # the deltas from the two distributions
    dist_1_deltas = input_data['dist_1_deltas']  # (500,)
    dist_2_deltas  = input_data['dist_2_deltas' ]  # (500,)

    # generated the deltas weighted with the static alpha
    weighted_deltas = torch.tensor((1 - alpha) * dist_1_deltas + alpha * dist_2_deltas)  # (500,)

    # create the synthetic time series data, X and Y will be directly pulled from this tensor
    cumsum_data = torch.cumsum(weighted_deltas, dim=0)  # (500,)

    # calculate the number of training data points we can use
    training_length = len(cumsum_data) - x_window_size - y_window_size

    # Create indices for extracting windows
    indices = torch.arange(x_window_size).view(1, -1) + torch.arange(training_length).view(-1, 1)    # (445, 50)

    # Use advanced indexing to create train_x
    train_x = cumsum_data[indices]  # (445, 50)

    # Create indices for extracting windows
    indices = torch.arange(y_window_size).view(1, -1) + torch.arange(training_length).view(-1, 1)    # (445, 5)
    indices = torch.add(indices, x_window_size)

    # Use advanced indexing to create train_x
    train_y = cumsum_data[indices]  # (445, 50)

    # get the prediction deltas # TODO check all this!!!!!
    dist_1_prediction_deltas = input_data['dist_1_prediction_deltas']  # (500,)
    prediction_1 = torch.tensor(dist_1_prediction_deltas)[indices]
    prediction_1 = torch.cumsum(prediction_1, dim=1)
    prediction_1 = prediction_1 + torch.select(train_x, 1, -1).view(-1, 1)   # (445,)

    dist_2_prediction_deltas = input_data['dist_2_prediction_deltas']  # (500,)
    prediction_2 = torch.tensor(dist_2_prediction_deltas)[indices]
    prediction_2 = torch.cumsum(prediction_2, dim=1)
    prediction_2 = prediction_1 + torch.select(train_x, 1, -1).view(-1, 1)   # (445,)

    np.savez('dataset/synthesized_data.npz',
             train_x=train_x,
             train_y=train_y,
             prediction_1=prediction_1,
             prediction_2=prediction_2)


if __name__ == "__main__":
    synthesize_deltas()
    synthesize_static_alpha_data()
