import numpy as np
import torch

from base_model_full_data import train_and_predict_model
from base_model_door_open import train_and_predict_model_with_door_open
from base_model_utils import import_scaled_data, get_sample_indices, unify_data_base_models
from parse_sensor_data import plot_test_results, plot_test_mape_loss, plot_test_mae_loss
from rlmc import run_rlmc

FEATURE_LENGTH = 7


def run_rlmc_test(seedling):

    with open('output.txt', 'a') as f:
        f.write(f"\n\nseedling: {seedling}\n")

    bm_1 = train_and_predict_model_with_door_open(seed=seedling+100)
    bm_2 = train_and_predict_model(seed=seedling+200)
    data = import_scaled_data()

    # TODO figure out how to address predictions being shorter than actual length
    print(bm_1.shape, bm_2.shape, data.shape)

    data = import_scaled_data()

    feature_indices, label_indices = get_sample_indices(data_length=len(data),
                                                        x_window_size=FEATURE_LENGTH,
                                                        y_window_size=1)

    X = data[feature_indices]
    y = data[label_indices]

    X = X[:, :, np.newaxis]

    # HOLY SHIT BIG DIFFERENCE
    # Model 1 MAE error: 0.51866
    # Model 1 MAPE error: 14.22073
    # Model 2 MAE error: 0.24918
    # Model 2 MAPE error: 5.64110
    # test_mae_loss: 0.23774
    # test_mape_loss: 5.38738

    # this the leading FEATURE length data was removed because predictions were shorter
    # Model 1 MAE error: 0.51866
    # Model 1 MAPE error: 14.22073
    # Model 2 MAE error: 0.24918
    # Model 2 MAPE error: 5.64110
    # test_mae_loss: 0.20633
    # test_mape_loss: 4.60492

    # np.save('bm_1.npy', bm_1)
    # np.save('bm_2.npy', bm_2)
    # np.save('X.npy', X)
    # np.save('Y.npy', y)

    unify_data_base_models(X, y, bm_1, bm_2)

    # run RL agent
    torch.manual_seed(seedling+300)
    run_rlmc(False, True, True, True, 0.5)


if __name__ == "__main__":

    run_rlmc_test(seedling=5)

