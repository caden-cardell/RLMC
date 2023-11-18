import torch
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from utils import compute_mape_error, compute_mae_error
from run_rlmc import run_rlmc


def get_sample_indices(data_length, x_window_size=120, y_window_size=24):

    sample_length = data_length - x_window_size - y_window_size

    # create feature indices
    feature_indices = torch.arange(x_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)

    # create label indices
    label_indices = torch.arange(y_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)
    label_indices = torch.add(label_indices, x_window_size)

    return feature_indices, label_indices


def load_data_from_csv(input_filepath):

    def extract_last_float(line):
        # Extract all float numbers from the line
        floats = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        # Return the last float if any, else return None
        return floats[-1] if floats else None

    data = []
    with open(input_filepath, 'r') as file:
        for line in file:
            last_float = float(extract_last_float(line))
            data.append(last_float)

    return data


def synthesize_predictions_from_data(input_filepath):
    # import data
    data = np.array(load_data_from_csv(input_filepath))
    data_length = len(data)

    # generate prediction error
    start_mean, end_mean = 0, 0
    start_std, end_std = .1, .1

    # Create tensors for linearly interpolated mean and standard deviation
    means = torch.linspace(start_mean, end_mean, data_length)
    stds = torch.linspace(start_std, end_std, data_length)

    # Sample from the normal distribution for each index
    dist_1_prediction_deltas = torch.stack([torch.normal(mean, std) for mean, std in zip(means, stds)])

    # Define parameters for the sine wave
    amplitude = 1  # Amplitude of the sine wave
    frequency = 14 * math.pi / data_length  # Frequency - one full cycle over the number of points

    # Create a tensor for the x values
    x_values = torch.linspace(0, data_length - 1, data_length)

    # Calculate the sine wave for the means
    means = amplitude * torch.sin(frequency * x_values)

    # Define standard deviation range
    start_std, end_std = .1, .1

    # Create tensors for linearly interpolated standard deviation
    stds = torch.linspace(start_std, end_std, data_length)

    # Sample from the normal distribution for each index
    dist_2_prediction_deltas = torch.stack([torch.normal(mean, std) for mean, std in zip(means, stds)])

    dist_1_prediction = dist_1_prediction_deltas + data
    dist_2_prediction = dist_2_prediction_deltas + data

    return np.column_stack((dist_1_prediction, dist_2_prediction))


def synthesize_homogenous_predictions_from_data(input_filepath):
    # import data
    data = np.array(load_data_from_csv(input_filepath))
    data_length = len(data)

    # generate prediction error
    start_mean, end_mean = 0, 0
    start_std, end_std = .1, .1

    # Create tensors for linearly interpolated mean and standard deviation
    means = torch.linspace(start_mean, end_mean, data_length)
    stds = torch.linspace(start_std, end_std, data_length)

    # Sample from the normal distribution for each index
    dist_1_prediction_deltas = torch.stack([torch.normal(mean, std) for mean, std in zip(means, stds)])

    # Sample from the normal distribution for each index
    dist_2_prediction_deltas = torch.stack([torch.normal(mean, std) for mean, std in zip(means, stds)])

    dist_1_prediction = dist_1_prediction_deltas + data
    dist_2_prediction = dist_2_prediction_deltas + data

    return np.column_stack((dist_1_prediction, dist_2_prediction))


def create_samples_from_data(input_filepath, x_window_size=120, y_window_size=24):
    # load data
    data = np.array(load_data_from_csv(input_filepath))

    # calculate number of data points
    data_length = len(data)

    # create indices
    feature_indices, label_indices = get_sample_indices(data_length,
                                                        x_window_size=x_window_size,
                                                        y_window_size=y_window_size)

    # use advanced indexing to create features
    feature_samples = data[feature_indices][:, :, np.newaxis]  # add dimension because this model works with multi-feature data

    # use advanced indexing to create labels
    label_samples = data[label_indices]

    return feature_samples, label_samples


def create_prediction_samples_from_data(input_filepath, x_window_size=120, y_window_size=24):
    # get base model predictions
    base_model_predictions = synthesize_homogenous_predictions_from_data(input_filepath)

    # get important dimensions
    number_of_base_models = base_model_predictions.shape[1]  # number of base models
    data_length = len(base_model_predictions)

    # get the indices for training samples
    _, label_indices = get_sample_indices(data_length, x_window_size=x_window_size, y_window_size=y_window_size)

    # get samples from base models
    merge_base_models_samples = []
    for base_model_num in range(number_of_base_models):
        bm = base_model_predictions[:, base_model_num]
        bm = bm[label_indices]
        bm = np.expand_dims(bm, axis=1)
        merge_base_models_samples.append(bm)
    base_model_samples = np.concatenate(merge_base_models_samples, axis=1)

    return base_model_samples


def plot_data_vs_predictions(data, base_model_predictions):

    # plotting
    n = base_model_predictions.shape[1]  # number of base models
    for base_model_num in range(n):
        plt.plot(base_model_predictions[:, base_model_num], label=f"base model {base_model_num}")
    plt.plot(data, label="data", color='red')

    # labeling
    plt.xlabel('15 minute time steps')
    plt.ylabel('Degrees F')
    plt.title('Synthetic Model Predictions')
    plt.legend()

    # Display the plot
    plt.show()


def create_synthetic_model_prediction_plots():
    # get data
    data = np.array(load_data_from_csv(input_filepath='data.csv'))

    # synthesis the models' predictions
    base_model_predictions = synthesize_homogenous_predictions_from_data(input_filepath='data.csv')

    # plot
    plot_data_vs_predictions(data=data, base_model_predictions=base_model_predictions)


def unify_sample_data(input_filepath='data.csv', x_window_size=120, y_window_size=24):

    feature_samples, label_samples = create_samples_from_data(input_filepath,
                                                              x_window_size=x_window_size,
                                                              y_window_size=y_window_size)
    base_model_samples = create_prediction_samples_from_data(input_filepath,
                                                             x_window_size=x_window_size,
                                                             y_window_size=y_window_size)

    L = len(feature_samples) // 8

    test_X = feature_samples[-L:]
    valid_X = feature_samples[-(2 * L):-L]
    train_X = feature_samples[:-(2 * L)]

    test_y = label_samples[-L:]
    valid_y = label_samples[-(2 * L):-L]
    train_y = label_samples[:-(2 * L)]

    test_preds = base_model_samples[-L:]
    valid_preds = base_model_samples[-(2 * L):-L]
    train_preds = base_model_samples[:-(2 * L)]

    test_error_df  = compute_mape_error(test_y , test_preds)
    valid_error_df = compute_mape_error(valid_y, valid_preds)
    train_error_df = compute_mape_error(train_y, train_preds)

    np.save('dataset/bm_test_preds.npy', test_preds)
    np.save('dataset/bm_valid_preds.npy', valid_preds)
    np.save('dataset/bm_train_preds.npy', train_preds)

    np.savez('dataset/input.npz',
             test_X=test_X,
             valid_X=valid_X,
             train_X=train_X,
             test_y=test_y,
             valid_y=valid_y,
             train_y=train_y,
             test_error=test_error_df,
             valid_error=valid_error_df,
             train_error=train_error_df
            )


def plot_test_results():

    input_data = np.load('dataset/test_results.npz')

    test_bm_preds = input_data['test_bm_preds']
    rl_preds  = input_data['rl_preds'][:, 0]
    test_y = input_data['test_y'][:, 0]

    pass

    # plotting
    n = test_bm_preds.shape[1]  # number of base models
    for base_model_num in range(n):
        plt.plot(test_bm_preds[:, base_model_num, 0], label=f"Base model {base_model_num}")

    plt.plot(rl_preds, label="RL agent predictions", color='blue')
    plt.plot(test_y, label="True data", color='red')

    # labeling
    plt.xlabel('15 minute time steps')
    plt.ylabel('Degrees F')
    plt.title('Synthetic Model Test Results')
    plt.legend()

    # Display the plot
    plt.show()


def plot_test_mape_loss():

    # get data
    input_data = np.load('dataset/test_results.npz')
    test_bm_preds = input_data['test_bm_preds']
    rl_preds  = input_data['rl_preds']
    test_y = input_data['test_y']

    # plotting
    n = test_bm_preds.shape[1]  # number of base models
    for base_model_num in range(n):

        # get model
        bm = test_bm_preds[:, base_model_num, :]

        # get loss
        mape_loss = compute_mape_error(test_y, bm[:, np.newaxis, :])

        plt.plot(mape_loss, label=f"Base model {base_model_num} error")

    mape_loss = compute_mape_error(test_y, rl_preds[:, np.newaxis, :])
    plt.plot(mape_loss, label="RL agent error", color='red')

    # labeling
    plt.xlabel('15 minute time steps')
    plt.ylabel('Error')
    plt.title('MAPE Error')
    plt.legend()

    # Display the plot
    plt.show()


def plot_test_mae_loss():

    # get data
    input_data = np.load('dataset/test_results.npz')
    test_bm_preds = input_data['test_bm_preds']
    rl_preds  = input_data['rl_preds']
    test_y = input_data['test_y']

    # plotting
    n = test_bm_preds.shape[1]  # number of base models
    for base_model_num in range(n):

        # get model
        bm = test_bm_preds[:, base_model_num, :]

        # get loss
        mae_loss = compute_mae_error(test_y, bm[:, np.newaxis, :])

        plt.plot(mae_loss, label=f"Base model {base_model_num} error")

    mae_loss = compute_mae_error(test_y, rl_preds[:, np.newaxis, :])
    plt.plot(mae_loss, label="RL agent error", color='red')

    # labeling
    plt.xlabel('15 minute time steps')
    plt.ylabel('Error')
    plt.title('MAE Error')
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == '__main__':

    # create data
    unify_sample_data(input_filepath='data.csv', x_window_size=120, y_window_size=24)

    # plot synthetic data
    create_synthetic_model_prediction_plots()

    # run RL agent
    run_rlmc(False, True, True, True, 0.5)

    # plot results
    plot_test_results()

    # plot error
    plot_test_mape_loss()
    plot_test_mae_loss()
