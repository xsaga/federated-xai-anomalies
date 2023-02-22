"""Compute SHAP values across multiple clients in a federated learning (FL) setting. Multiprocessing version."""

import argparse
import pickle
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import shap

from matplotlib import pyplot as plt
import seaborn as sns

from fkms.algorithms import kfed


def reconstruction_error(model, loss_function, samples):
    """Apply the model prediction to a list of samples."""
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error.numpy()


def model_predict(X):
    return reconstruction_error(model, loss_func, torch.from_numpy(X))


def federated_baseline(client_data: List[np.ndarray], num_instances: int, column_names):
    local_estimates, global_centers, global_counts = kfed(client_data, num_instances, num_instances, round_values=True)
    return shap.utils._legacy.DenseData(global_centers, column_names, None, global_counts.astype(np.float64))


parser = argparse.ArgumentParser(description="Compute SHAP values across multiple clients in a FL setting.")
parser.add_argument("--method", choices=["packet", "flow"], required=True,
                    help="The method type to apply the correct model and processing to the data.")
parser.add_argument("-m", "--model", type=lambda p: Path(p).absolute(), required=True,
                    help="Trained model.")
parser.add_argument("--dimensions", required=True, type=int,
                    help="Number of input features for the model.")
parser.add_argument("-k", "--kmeans-sample", required=True, type=int, default=5,
                    help="the k for k-means when computing the federated SHAP baseline.")
parser.add_argument("-s", "--sample-frac", required=False, type=float,
                    help="Optional. Compute SHAP on a fraction of the anomalies.")
parser.add_argument("-v", "--validation", type=lambda p: Path(p).absolute(), nargs="+", required=True,
                    help="List of validation datasets. One for each FL client.")
parser.add_argument("-t", "--test", type=lambda p: Path(p).absolute(), nargs="+", required=True,
                    help="List of test datasets. One for each FL client.")
parser.add_argument("-o", "--output", type=lambda p: Path(p).absolute(), required=False,
                    help="Output filename of the serialized results.")

args = parser.parse_args()

assert len(args.validation) == len(args.test), "Clients must have the same number of validation and test datasets"

num_clients = len(args.validation)
experiment_timestamp = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")

torch.set_num_threads(2)

#####################################
# Conditional imports and functions #
#####################################

if args.method == "packet":
    from feature_extractor import port_hierarchy_map_iot, preprocess_dataframe
    from model_ae import Autoencoder

    def load_data(X: Path) -> pd.DataFrame:
        return preprocess_dataframe(pd.read_pickle(X), port_mapping=port_hierarchy_map_iot)

    def compute_threshold(model, loss_function, X: pd.DataFrame) -> float:
        return np.max(reconstruction_error(model, loss_function, torch.from_numpy(X.astype(np.float32).to_numpy())))

    def compute_reconstruction_error(model, loss_function, X: pd.DataFrame) -> np.ndarray:
        return reconstruction_error(model, loss_function, torch.from_numpy(X.astype(np.float32).to_numpy()))

if args.method == "flow":
    from nbaiot_process import Autoencoder
    from nbaiot_process import compute_threshold as nbaiot_compute_threshold

    def load_data(X: Path) -> pd.DataFrame:
        return pd.read_pickle(X)

    def compute_threshold(model, loss_function, X: pd.DataFrame) -> float:
        return nbaiot_compute_threshold(reconstruction_error(model, loss_function, torch.from_numpy(X.to_numpy())))

    def compute_reconstruction_error(model, loss_function, X: pd.DataFrame) -> np.ndarray:
        return reconstruction_error(model, loss_function, torch.from_numpy(X.to_numpy()))

######################
# Load trained model #
######################

print("Loading trained model")

model = Autoencoder(args.dimensions)
model.load_state_dict(torch.load(args.model)["state_dict"])
loss_func = F.mse_loss

####################
# Load client data #
####################

# load validation datasets
print("Loading validation datasets")

clients_valid_dfs = []
for x in args.validation:
    x_df = load_data(x)
    if "timestamp" in x_df.columns:
        clients_valid_dfs.append(x_df.drop(columns=["timestamp"]))
    else:
        clients_valid_dfs.append(x_df)

# compute anomaly detection thresholds in each client
print("Computing anomaly thresholds")
clients_thresholds = list(map(lambda x: compute_threshold(model, loss_func, x),
                              clients_valid_dfs))

# load test datasets
clients_test_dfs = []
clients_test_timestamps = []
for i, x in enumerate(args.test):
    x_df = load_data(x)
    print(f"Client #{i} test dataset shape: {x_df.shape}")
    if "timestamp" in x_df:
        clients_test_dfs.append(x_df.drop(columns=["timestamp"]))
        clients_test_timestamps.append(x_df["timestamp"].values)
    else:
        clients_test_dfs.append(x_df)
        clients_test_timestamps.append(np.arange(x_df.shape[0]))

# apply model for each dataframe
clients_test_results = list(map(lambda x: compute_reconstruction_error(model, loss_func, x),
                                clients_test_dfs))

# visualize
fig, ax = plt.subplots(nrows=int(np.ceil(num_clients / 2)), ncols=2)
for i, x in enumerate(clients_test_results):
    sns.scatterplot(x=clients_test_timestamps[i], y=clients_test_results[i], linewidth=0, s=12, alpha=0.5, ax=ax.flatten()[i], rasterized=True)
    ax.flatten()[i].axhline(y=clients_thresholds[i], c="k")
    ax.flatten()[i].set_title(f"Client #{i}")
fig.tight_layout()
plt.show()

###############
# SHAP values #
###############

# SHAP baseline in a federated learning setting using the validation dataset
feat_names = clients_valid_dfs[0].columns

print("Computing federated SHAP baseline")
shap_baseline = federated_baseline(list(map(lambda x: x.astype(np.float32).to_numpy(), clients_valid_dfs)),
                                   num_instances=args.kmeans_sample,
                                   column_names=feat_names)

# SHAP explainer objects for each client
clients_explainers = []
for i in range(num_clients):
    clients_explainers.append(shap.KernelExplainer(model_predict, shap_baseline))

# filter anomalies in the test dataset
clients_anomalies = []
for i in range(num_clients):
    anom_df = clients_test_dfs[i][clients_test_results[i] > clients_thresholds[i]]
    print(f"Client #{i} anomalies shape: {anom_df.shape}")
    # optional subsampling, need for speed
    if args.sample_frac:
        anom_df = anom_df.sample(frac=args.sample_frac, replace=False, random_state=1, ignore_index=False).sort_index()
        print(f"\tsubsampled (fraction {args.sample_frac}) to: {anom_df.shape}")
    clients_anomalies.append(anom_df)

# compute SHAP values
# clients_anomalies_shap = []
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore")
#     for i in range(num_clients):
#         print(f"Computing SHAP values for client #{i}")
#         shap_i = clients_explainers[i].shap_values(clients_anomalies[i].astype(np.float32))
#         clients_anomalies_shap.append(shap_i)

def compute_shap_client(explainer_anomalies_pair):
    # set the torch number of threads also here, so that each
    # process spawned by the multiprocessing module limits the
    # number of threads used by pytorch. Avoids reduced performance
    # due to thread contention.
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        explainer, anomalies = explainer_anomalies_pair
        shap_i = explainer.shap_values(anomalies.astype(np.float32))
    return shap_i

with Pool(len(args.test)) as p:
    clients_anomalies_shap = p.map(compute_shap_client, list(zip(clients_explainers, clients_anomalies)))

# serialize results
output_file = args.output if args.output else f"shap_{args.method}_{num_clients}clients_{args.kmeans_sample}baseline_{str(args.sample_frac)+'frac' if args.sample_frac else 'all'}_anom_{experiment_timestamp}.pickle"

results_container = {"clients_thresholds": clients_thresholds,
                     "clients_validation_data_path": args.validation,
                     "clients_test_data_path": args.test,
                     "clients_test_dfs": clients_test_dfs,
                     "clients_test_timestamps": clients_test_timestamps,
                     "clients_test_results": clients_test_results,
                     "shap_baseline": shap_baseline,
                     "clients_explainers": clients_explainers,
                     "subsampled_anomalies_frac": args.sample_frac,
                     "clients_anomalies": clients_anomalies,
                     "clients_anomalies_shap": clients_anomalies_shap}

with open(output_file, "wb") as f:
    pickle.dump(results_container, f)

print("Results saved to: ", output_file)
