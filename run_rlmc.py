import os

import numpy as np
import torch

from base_model_full_data import train_and_predict_model
from base_model_door_open import train_and_predict_model_with_door_open
from base_model_utils import import_scaled_data, generate_sequence_sampling_indices, unify_data_base_models
from rlmc import run_rlmc
import matplotlib.pyplot as plt

FEATURE_LENGTH = 16


def run_rlmc_test(seedling): # TODO better name

    # get every single
    bm_1 = train_and_predict_model_with_door_open(seed=(1000+seedling))
    bm_2 = train_and_predict_model(seed=(2000+seedling))

    data = import_scaled_data()

    feature_indices, label_indices = generate_sequence_sampling_indices(total_sequence_length=len(data),
                                                                        input_sequence_length=FEATURE_LENGTH,
                                                                        output_sequence_length=1)

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
        torch.manual_seed((3000+seedling))

    # these parameters are the defaults in the original code
    preds, weights = run_rlmc(False, True, True, True, 0.5)

    # TODO save rlmc test predictions and weights and the seed used to generate them

    # plot bm_1, bm_2, preds, weights, y
    plt.plot(y, label="data", color='red')
    plt.plot(preds, label="preds", color='blue')
    plt.plot(bm_1, label="bm_1", color='orange')
    plt.plot(bm_2, label="bm_2", color='green')
    plt.plot(weights[:, 0], label="weights", color='gray')
    plt.legend()

    # Display the plot
    plt.show()

    np.savez(f'output/data_{seedling}.npz',
             preds=preds,
             weights=weights,
             bm_1=bm_1,
             bm_2=bm_2
            )


if __name__ == "__main__":

    for i in range(40):
        run_rlmc_test(seedling=(i))


