import os

import numpy as np
import torch

from base_model_full_data import train_and_predict_model
from base_model_door_open import train_and_predict_model_with_door_open
from base_model_utils import import_scaled_data, get_sample_indices, unify_data_base_models
from parse_sensor_data import plot_test_results, plot_test_mape_loss, plot_test_mae_loss
from rlmc import run_rlmc

FEATURE_LENGTH = 16


def run_rlmc_test(seedling):

    with open('output.txt', 'a') as f:
        f.write(f"\n\nseedling: {seedling}\n")

    bm_1 = train_and_predict_model_with_door_open(seed=3)
    bm_2 = train_and_predict_model(seed=3)

    data = import_scaled_data()

    feature_indices, label_indices = get_sample_indices(data_length=len(data),
                                                        x_window_size=FEATURE_LENGTH,
                                                        y_window_size=1)

    X = data[feature_indices]
    y = data[label_indices]

    X = X[:, :, np.newaxis]

    unify_data_base_models(X, y, [bm_1, bm_2])

    try:
        os.remove("./dataset/batch_buffer.csv")
        print("Buffer deleted.")
    except FileNotFoundError as e:
        print("Buffer already deleted.")

    # run RL agent
    # torch.manual_seed(seedling+300)
    run_rlmc(False, True, True, True, 0.5)


if __name__ == "__main__":

    # Model 0 MAE error: 0.52117
    # Model 0 MAPE error: 14.33724
    # Model 1 MAE error: 0.46180
    # Model 1 MAPE error: 10.75681

    for i in range(1):
        run_rlmc_test(seedling=(7500+(i+1)))
        plot_test_results()


