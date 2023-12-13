# coding: utf-8

# In[57]:


import numpy as np
import matplotlib.pyplot as plt
from base_model_utils import import_scaled_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import os


# In[2]:


def fahrenheit_to_celcius(list_of_fahrenheit):
    list_of_celcius = list_of_fahrenheit - 32
    list_of_celcius = list_of_celcius / 1.8
    return list_of_celcius


def fahrenheit_to_kelvin(list_of_fahrenheit):
    list_of_celcius = list_of_fahrenheit - 32
    list_of_celcius = list_of_celcius / 1.8
    return list_of_celcius + 273.15


# In[24]:


from base_model_utils import load_data_from_csv


def evaluate_outputs_main():

    # get data
    unscaled_data = load_data_from_csv("75e9_2023_09_07_11_02_to_2023_09_13_11_02.csv")
    unscaled_data = unscaled_data[:, np.newaxis]

    # find rescaling function to convert from -1, 1 to original degrees F
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit_transform(unscaled_data)

    # for each file in /output
    directory = "./output"
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            # Load the .npy file
            file_path = os.path.join(directory, filename)
            scaled_data = np.load(file_path)

            # preds
            preds = scaled_data["preds"]
            preds = scaler.inverse_transform(preds)
            preds = fahrenheit_to_celcius(preds)

            # weights
            weights = scaled_data["weights"]

            # bm_1
            bm_1 = scaled_data["bm_1"]
            bm_1 = scaler.inverse_transform(bm_1)
            bm_1 = fahrenheit_to_celcius(bm_1)

            # bm_2
            bm_2 = scaled_data["bm_2"]
            bm_2 = scaler.inverse_transform(bm_2)
            bm_2 = fahrenheit_to_celcius(bm_2)

            # print(filename, preds)

            # # save to /rescaled_output
            file_path = os.path.join("./scaled-output", filename)
            np.savez(file_path,
                     preds=preds,
                     weights=weights,
                     bm_1=bm_1,
                     bm_2=bm_2
                     )


    # In[25]:


    def get_data(seedling):
        # import reading data
        data = load_data_from_csv("75e9_2023_09_07_11_02_to_2023_09_13_11_02.csv")
        data = fahrenheit_to_celcius(data)

        # import prediction data
        prediction_data = np.load(f'scaled-output/data_{seedling}.npz')
        preds = prediction_data['preds']
        weights = prediction_data['weights']
        bm_1 = prediction_data['bm_1']
        bm_2 = prediction_data['bm_2']

        # fix length
        length = min([len(l) for l in [data, preds, weights, bm_1, bm_2]])
        length = length - 2993
        data = data[-length:]
        preds = preds[-length:, 0]
        weights = weights[-length:]
        bm_2 = bm_2[-length:, 0]
        bm_1 = bm_1[-length:, 0]

        return data, preds, weights, bm_1, bm_2


    # In[26]:


    bm_1_total_data = []
    bm_2_total_data = []
    pred_total_data = []
    pred_control_total_data = []

    for i in range(20):
        # get the base model data from the first 20
        data, preds, weights, bm_1, bm_2 = get_data(i)
        bm_1_total_data.append(bm_1)
        bm_2_total_data.append(bm_2)
        pred_control_total_data.append(preds)

        # get the meta model predictions from the second 20
        data, preds, weights, bm_1, bm_2 = get_data(i + 20)
        pred_total_data.append(preds)

    bm_1_total_data = np.array(bm_1_total_data)
    bm_2_total_data = np.array(bm_2_total_data)
    pred_total_data = np.array(pred_total_data)
    pred_control_total_data = np.array(pred_control_total_data)

    bm_1_total_data.shape, bm_2_total_data.shape, pred_total_data.shape


    # In[58]:


    def graph_multi_hist_across_all_data():
        plt.figure(figsize=(12, 8))

        cum_mae_bm_1 = []
        cum_mae_bm_2 = []
        cum_mae_preds = []

        cum_mape_bm_1 = []
        cum_mape_bm_2 = []
        cum_mape_preds = []

        for idx in range(20):
            # use first 20 for base models
            data, preds, weights, bm_1, bm_2 = get_data(idx)

            # Calculate errors
            mae = mean_absolute_error(data, bm_1)
            cum_mae_bm_1.append(mae)
            mape = mean_absolute_percentage_error(data, bm_1)
            cum_mape_bm_1.append(mape)

            mae = mean_absolute_error(data, bm_2)
            cum_mae_bm_2.append(mae)
            mape = mean_absolute_percentage_error(data, bm_2)
            cum_mape_bm_2.append(mape)

            # use second 20 for meta preds
            data, preds, weights, bm_1, bm_2 = get_data(idx + 20)

            mae = mean_absolute_error(data, preds)
            cum_mae_preds.append(mae)
            mape = mean_absolute_percentage_error(data, preds)
            cum_mape_preds.append(mape)

        # Determine the common range for all histograms
        all_data = np.concatenate((cum_mae_bm_1, cum_mae_bm_2, cum_mae_preds))
        min_bin = np.min(all_data)
        max_bin = np.max(all_data)

        # Define the common bin edges
        bins = np.linspace(min_bin, max_bin, 40)

        # Histogram of Forecast Errors
        plt.hist(cum_mae_bm_1, bins=bins, label="Base Model 1", edgecolor='black', alpha=0.5)
        plt.hist(cum_mae_bm_2, bins=bins, label="Base Model 2", edgecolor='black', alpha=0.5)
        plt.hist(cum_mae_preds, bins=bins, label="RL Agent Predictions", edgecolor='black', alpha=0.5)

        plt.title('Histogram of Forecast MAE')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        # plt.show()
        plt.savefig('histogram_of_forecast_mae.png', dpi=300, format='png', bbox_inches='tight')

        print(f"Base Model 1 Total MAE {np.mean(cum_mae_bm_1)}")
        print(f"Base Model 2 Total MAE {np.mean(cum_mae_bm_2)}")
        print(f"RL Agent Predictions Total MAE {np.mean(cum_mae_preds)}")

        print(f"Base Model 1 Total MAPE {np.mean(cum_mape_bm_1)}")
        print(f"Base Model 2 Total MAPE {np.mean(cum_mape_bm_2)}")
        print(f"RL Agent Predictions Total MAPE {np.mean(cum_mape_preds)}")


    # In[28]:


    def get_cv(all_data):
        coef_of_var = []

        for idx in range(1240):
            temp = all_data[:, idx]

            var = np.var(temp)
            std = np.std(temp)
            mean = np.mean(temp)

            cv = std / mean

            coef_of_var.append(cv)

        coef_of_var = np.array(coef_of_var)

        return coef_of_var


    # In[60]:


    def plot_data_directly():
        data, preds, weights, bm_1, bm_2 = get_data(2)

        fig, ax1 = plt.subplots(figsize=(16, 10))
        x = range(len(data))

        # labeling
        ax1.plot(x, data, label="True Data", color='black')
        ax1.plot(x, bm_1, label="Base Model 1", color='blue')
        ax1.plot(x, bm_2, label="Base Model 2", color='green')
        ax1.plot(x, preds, label="RL Agent Predictions", color='red')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Celcius')

        # Creating a second y-axis on the right and plotting the second dataset
        ax2 = ax1.twinx()
        ax2.plot(x, weights[:, 0], label="Model 1:Model 2", color='gray')
        ax2.set_ylabel('Weights')

        ax1_legend = ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # Place the second legend below the first
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.87), borderaxespad=0.)
        # plt.gca().add_artist(ax1_legend)  # Ensure first legend stays on top

        # Adjust the layout
        plt.subplots_adjust(right=0.75)  # Adjust as needed

        # Use tight layout to automatically adjust subplot params
        plt.tight_layout()

        # Display the plot
        plt.title('Model Prediction Comparison')

        # plt.show()
        # plt.savefig('model_prediction_comparison.png')

        plt.tight_layout()

        # Save the figure with a high resolution
        plt.savefig('model_prediction_comparison.png', dpi=300, format='png', bbox_inches='tight')


    # In[59]:


    # PLOT HISTOGRAMS OF MAE
    plot_data_directly()
    graph_multi_hist_across_all_data()

    # In[10]:


    # PLOT COEF OF VAR
    plt.figure(figsize=(12, 8))

    plt.plot(get_cv(bm_2_total_data), label="CV of bm_2 error", color='purple', alpha=0.75)
    plt.plot(get_cv(pred_total_data), label="CV of predicted error", color='black', alpha=0.9)
    plt.plot(get_cv(bm_1_total_data), label="CV of bm_1 error", color='green', alpha=0.75)
    plt.legend()
    plt.xlabel('time steps')
    plt.ylabel('Coef of Var')
    plt.title('Coef of Var across models')

    # Display the plot
    # plt.show()
    plt.savefig('coef_of_variability.png', dpi=300, format='png', bbox_inches='tight')

    # In[11]:


    cum_weights = []

    for idx in range(20):
        # use second 20 for meta preds
        _, _, weights, _, _ = get_data(idx + 20)
        cum_weights.append(weights[:, 0, 0])

    cum_weights = np.array(cum_weights)

    cum_weights_avg = np.mean(cum_weights, axis=0)

    plt.figure(figsize=(12, 8))

    length = len(cum_weights_avg) // 2

    plt.plot(cum_weights_avg[-length:], label="bm_1:bm_2", color='black')
    plt.plot(get_cv(cum_weights)[-length:], label="CV of bm_1:bm_2", color='red')
    plt.xlabel('time steps')
    plt.ylabel('Weight')
    plt.title('Base model weight ratio')

    # Display the plot
    # plt.show()
    plt.savefig('average_base_model_weight.png', dpi=300, format='png', bbox_inches='tight')

    # In[ ]:


if __name__ == '__main__':
    print("Generating graphs and calculating error")
    evaluate_outputs_main()

