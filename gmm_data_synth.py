import torch
import numpy as np
from utils import compute_mape_error


def synthesize_deltas(number_of_points=500):
    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Define the means and standard deviations
    dist_1_mean = 0
    dist_1_std_dev = 1

    dist_2_mean = 1
    dist_2_std_dev = 4

    # Define the size of the output tensor
    size = (number_of_points,)

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


def synthesize_static_alpha_data(x_window_size=50, y_window_size=5):

    # parameters
    alpha = 0.25

    # import data
    input_data = np.load('dataset/synthesized_deltas.npz')

    # the deltas from the two distributions
    dist_1_deltas = input_data['dist_1_deltas']  # (500,)
    dist_2_deltas  = input_data['dist_2_deltas']  # (500,)

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

    # Create indices y
    indices = torch.arange(y_window_size).view(1, -1) + torch.arange(training_length).view(-1, 1)    # (445, 5)
    indices = torch.add(indices, x_window_size)

    # Use advanced indexing to create train_x
    train_y = cumsum_data[indices]  # (445, 50)

    # get the prediction deltas # TODO check all this!!!!!
    dist_1_prediction_deltas = input_data['dist_1_prediction_deltas']  # (500,)
    prediction_1 = torch.tensor(dist_1_prediction_deltas)[indices]  # use the same indices as the ys but extract from the prediction deltas
    prediction_1 = torch.cumsum(prediction_1, dim=1)  # cumulative sum the deltas
    prediction_1 = prediction_1 + torch.select(train_x, 1, -1).view(-1, 1)   # (445,), take the last element of each x and add to each of the cumsums, this is the prediction

    dist_2_prediction_deltas = input_data['dist_2_prediction_deltas']  # (500,)
    prediction_2 = torch.tensor(dist_2_prediction_deltas)[indices]
    prediction_2 = torch.cumsum(prediction_2, dim=1)
    prediction_2 = prediction_1 + torch.select(train_x, 1, -1).view(-1, 1)   # (445,)

    np.savez('dataset/synthesized_data.npz',
             train_x=train_x,
             train_y=train_y,
             prediction_1=prediction_1,
             prediction_2=prediction_2)


def gmm_unify_input_data():

    # import data
    input_data = np.load('dataset/synthesized_data.npz')

    # separate data
    all_X = input_data['train_x'][:, :, np.newaxis]
    all_Y  = input_data['train_y']

    L = len(all_X) // 10

    train_X = all_X[:-(2*L)]
    test_X = all_X[-L:]
    valid_X = all_X[-(2*L):-L]

    train_y = all_Y[:-(2*L)]
    test_y = all_Y[-L:]
    valid_y = all_Y[-(2*L):-L]

    pred_1  = input_data['prediction_1']  # (500,)
    pred_2  = input_data['prediction_2']  # (500,)

    pred_1 = np.expand_dims(pred_1, axis=1)
    pred_2 = np.expand_dims(pred_2, axis=1)

    merge_data = [pred_1, pred_2]
    merge_data = np.concatenate(merge_data, axis=1)  # (62795, 9, 24)

    train_preds = merge_data[:-(2*L)]
    test_preds = merge_data[-L:]
    valid_preds = merge_data[-(2*L):-L]

    np.save('dataset/bm_train_preds.npy', train_preds)
    np.save('dataset/bm_valid_preds.npy', valid_preds)
    np.save('dataset/bm_test_preds.npy', test_preds)

    train_error_df = compute_mape_error(train_y, train_preds)
    valid_error_df = compute_mape_error(valid_y, valid_preds)
    test_error_df  = compute_mape_error(test_y , test_preds)

    np.savez('dataset/input.npz',
             train_X=train_X,
             valid_X=valid_X,
             test_X=test_X,
             train_y=train_y,
             valid_y=valid_y,
             test_y=test_y,
             train_error=train_error_df,
             valid_error=valid_error_df,
             test_error=test_error_df
            )


if __name__ == "__main__":
    # synthesize_deltas(number_of_points=5000)
    # synthesize_static_alpha_data(x_window_size=120, y_window_size=24)
    gmm_unify_input_data()