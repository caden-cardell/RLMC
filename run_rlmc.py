import os

import numpy as np
import torch

from base_model_full_data import train_and_predict_model
from base_model_door_open import train_and_predict_model_with_door_open
from base_model_utils import import_scaled_data, get_sample_indices, unify_data_base_models
from rlmc import run_rlmc

FEATURE_LENGTH = 16


def run_rlmc_test(seedling): # TODO better name

    # get every single
    bm_1 = train_and_predict_model_with_door_open(seed=seedling)
    bm_2 = train_and_predict_model(seed=seedling)

    data = import_scaled_data()

    feature_indices, label_indices = get_sample_indices(data_length=len(data),
                                                        x_window_size=FEATURE_LENGTH,
                                                        y_window_size=1)

    # TODO: save model predictions with the seed used to generate them

    X = data[feature_indices]
    y = data[label_indices]

    X = X[:, :, np.newaxis]

    # unify the data and save to files for the RLMC
    unify_data_base_models(X, y, [bm_1, bm_2])

    # delete the batch buffer so each test is independent
    try:
        os.remove("./dataset/batch_buffer.csv")
        print("Batch buffer deleted.")
    except FileNotFoundError as e:
        print("Batch buffer already deleted.")

    # run RL agent
    if seedling is not None:
        torch.manual_seed(seedling)

    # these parameters are the defaults in the original code
    run_rlmc(False, True, True, True, 0.5)

    # TODO save rlmc test predictions and weights and the seed used to generate them


if __name__ == "__main__":

    for i in range(1):
        run_rlmc_test(seedling=(7500+(i+1)))


