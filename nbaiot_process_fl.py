import argparse
import copy
import sys

from collections import OrderedDict

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import seaborn as sns


def read_nbaiot_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", header=0, dtype=np.float32)


def read_nbaiot_multiple_fromdir(path: str) -> List[Dict[str, pd.DataFrame]]:
    assert path.is_dir()
    csv_files = path.glob("*.csv")
    all_data = []
    for csv_file in csv_files:
        print(f"reading: {path.name} / {csv_file.name}")
        all_data.append({"name": f"{path.name.split('_')[0]}_{csv_file.stem}",
                         "data": read_nbaiot_data(csv_file)})
    return all_data


def split_benign_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """From the dataset description txt"""
    return train_test_split(data, train_size=2/3)


def compute_threshold(mse_values: np.ndarray) -> float:
    """From the paper"""
    sample_mean = np.mean(mse_values)
    sample_stdev = np.std(mse_values)
    return sample_mean + sample_stdev


class Autoencoder(nn.Module):
    """Layer number and sizes from the paper.
       They do not provide details about the activation function.
    """

    def __init__(self, num_input: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_input, int(num_input * 0.75)),
            nn.ReLU(),
            nn.Linear(int(num_input * 0.75), int(num_input * 0.5)),
            nn.ReLU(),
            nn.Linear(int(num_input * 0.5), int(num_input * 0.33)),
            nn.ReLU(),
            nn.Linear(int(num_input * 0.33), int(num_input * 0.25)),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(num_input * 0.25), int(num_input * 0.33)),
            nn.ReLU(),
            nn.Linear(int(num_input * 0.33), int(num_input * 0.5)),
            nn.ReLU(),
            nn.Linear(int(num_input * 0.5), int(num_input * 0.75)),
            nn.ReLU(),
            nn.Linear(int(num_input * 0.75), num_input),
            nn.ReLU()
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded

    def predict_samples(self, samples):
        with torch.no_grad():
            predictions = self.forward(samples)
            return torch.mean(F.mse_loss(samples, predictions, reduction="none"), dim=1).numpy()


def fit(model, optimizer, loss_function, epochs, train_generator) -> float:
    model.train()
    train_loss_acc = 0.0
    for epoch in range(epochs):
        train_loss_acc = 0.0
        for x_batch in train_generator:
            preds = model(x_batch)
            loss = loss_function(preds, x_batch)
            train_loss_acc += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch+1}/{epochs}: train loss {train_loss_acc/len(train_generator):.8f}")
    return train_loss_acc / len(train_generator)


def test(model, loss_function, valid_generator) -> float:
    valid_loss_acc = 0.0
    with torch.no_grad():
        model.eval()
        for x_batch in valid_generator:
            preds = model(x_batch)
            valid_loss_acc += loss_function(preds, x_batch).item()
    print(f"valid loss {valid_loss_acc/len(valid_generator):.8f}")
    return valid_loss_acc / len(valid_generator)


# Federated Learning
# From clustered-fl-anomaly-detection-iot/   fl_server.py
def fedavg(model_weights: List[List[torch.Tensor]], num_training_samples: List[int]) -> List[torch.Tensor]:
    """FedAvg model aggregation"""
    assert len(model_weights) == len(num_training_samples)
    new_weights = []
    total_training_samples = sum(num_training_samples)
    for layers in zip(*model_weights):
        weighted_layers = torch.stack([torch.mul(l, w) for l, w in zip(layers, num_training_samples)])
        averaged_layers = torch.div(torch.sum(weighted_layers, dim=0), total_training_samples)
        new_weights.append(averaged_layers)
    return new_weights


def main(nbaiot_data_basepath: Path, nbaiot_devices_names: List[str], fl_rounds: int, local_train_epochs: int, train_lr: float):
    DATA_BASEPATH = nbaiot_data_basepath
    GAFGYT_BASEDIR = Path("gafgyt_attacks.rar.d")
    MIRAI_BASEDIR = Path("mirai_attacks.rar.d")
    GLOBAL_MODEL_SERIALIZED_PATH = Path(f"fl_nbaiot_ae_{len(nbaiot_devices_names)}_clients.pt")
    SHOW = True

    torch.manual_seed(0)
    np.random.seed(0)

    all_df_benign = []
    all_ds_trn = []
    all_ds_opt = []
    for dev in nbaiot_devices_names:
        _df_benign = read_nbaiot_data(DATA_BASEPATH / dev / "benign_traffic.csv")
        _ds_trn, _ds_opt = split_benign_data(_df_benign)
        all_df_benign.append(_df_benign)
        all_ds_trn.append(_ds_trn)
        all_ds_opt.append(_ds_opt)

    # The paper does not discuss feature transformations
    # Train MinMaxScaler using all the data in the federated network
    # In production FL settings, each client will send to the server
    # the min, max values for each feature. The server computes the global
    # min and max to create the scaler object.
    global_transformer = preprocessing.MinMaxScaler().fit(pd.concat(all_ds_trn))

    all_ds_trn_scaled = []
    all_ds_opt_scaled = []
    for x in all_ds_trn:
        all_ds_trn_scaled.append(global_transformer.transform(x))

    for x in all_ds_opt:
        all_ds_opt_scaled.append(global_transformer.transform(x))

    # Data loaders
    all_trn_dl = []
    for x in all_ds_trn_scaled:
        # batch size ?
        all_trn_dl.append(DataLoader(torch.from_numpy(x), batch_size=16))

    global_model = Autoencoder(115)

    if GLOBAL_MODEL_SERIALIZED_PATH.is_file():
        print("Loading trained global model.")
        global_model.load_state_dict(torch.load(GLOBAL_MODEL_SERIALIZED_PATH))
    else:
        print("Training global model.")

        for round in range(fl_rounds):
            print(f"----- round {round} -----")

            #--- copy the global model into each client
            local_models = []
            for dev in nbaiot_devices_names:
                local_models.append(copy.deepcopy(global_model))

            #--- local training of the local models
            for i, x in enumerate(all_trn_dl):
                print(f"\t--- client {i} ---")
                # They provide info for lr and epochs, but not for the optimizer algorithm
                opt = optim.Adam(local_models[i].parameters(), lr=train_lr, weight_decay=1e-5)
                loss_func = F.mse_loss
                fit(local_models[i], optimizer=opt, loss_function=loss_func, epochs=local_train_epochs, train_generator=x)

            #--- federated averaging
            all_local_models = [list(x.state_dict().values()) for x in local_models]
            new_global_model = fedavg(all_local_models, np.ones(len(local_models), dtype=int))

            # Assertion when using only one device (i.e., non FL version)
            # for iii in range(len(new_global_model)):
            #     assert torch.allclose(new_global_model[iii], all_local_models[0][iii])

            global_model.load_state_dict(OrderedDict(zip(global_model.state_dict().keys(), new_global_model)))

        print(f"Serializing trained global model to {GLOBAL_MODEL_SERIALIZED_PATH}")
        torch.save(global_model.state_dict(), GLOBAL_MODEL_SERIALIZED_PATH)

    # compute anomaly threshold
    all_threshold = []
    for i, x in enumerate(all_ds_opt_scaled):
        _mse_ds_opt = global_model.predict_samples(torch.from_numpy(x))
        threshold = compute_threshold(_mse_ds_opt)
        all_threshold.append(threshold)
        print(f"Anomaly threshold client {i} = {threshold}")

        # visualize
        fig, ax = plt.subplots()
        sns.scatterplot(x=np.arange(_mse_ds_opt.shape[0]), y=_mse_ds_opt, linewidth=0, ax=ax)
        ax.axhline(y=threshold, c="k")
        ax.set_title("MSE AE(dataset optim)")
        ax.set_yscale("log")
        if SHOW:
            plt.show()
        else:
            fig.savefig(f"1_mse_dataset_optim_fl_client_{i}.png", format="png")

    # load evaluation data
    for dev_idx, dev in enumerate(nbaiot_devices_names):
        data_mal = read_nbaiot_multiple_fromdir(DATA_BASEPATH / dev / GAFGYT_BASEDIR)

        # some devices do not have Mirai data
        if (DATA_BASEPATH / dev / MIRAI_BASEDIR).is_dir():
            data_mal.extend(read_nbaiot_multiple_fromdir(DATA_BASEPATH / dev / MIRAI_BASEDIR))

        for dm in data_mal:
            dm["data"] = global_transformer.transform(dm["data"])

        # visualize
        fig, ax = plt.subplots(nrows=int(np.ceil((len(data_mal)) / (np.ceil(np.sqrt(len(data_mal)))))),
                               ncols=int(np.ceil(np.sqrt(len(data_mal)))))
        for i, dm in enumerate(data_mal):
            sns.scatterplot(x=np.arange(dm["data"].shape[0]), y=global_model.predict_samples(torch.from_numpy(dm["data"])), linewidth=0, ax=ax.flatten()[i])
            ax.flatten()[i].axhline(y=all_threshold[dev_idx], c="k")
            ax.flatten()[i].set_yscale("log")
            ax.flatten()[i].set_title(dm["name"])
        if SHOW:
            plt.show()
        else:
            fig.savefig(f"2_mse_attacks_fl_client_{dev_idx}.png", format="png")

        # generate labels
        label_map = {x["name"]: i for i, x in enumerate(data_mal)}
        data_labels = []
        for dm in data_mal:
            lbl = dm["name"]
            df_l = pd.DataFrame({"label": np.full(dm["data"].shape[0], label_map[lbl], dtype=np.int32),
                                 "type": np.full(dm["data"].shape[0], lbl)})
            data_labels.append(df_l)

        # serialize
        # model as tar
        chkpt = {"state_dict": global_model.state_dict(),
                "model_hash": None,
                "local_epochs": None,
                "loss": None,
                "train_loss": None,
                "num_samples": None}
        torch.save(chkpt, f"data/fl_{dev}_nbaiot_ae.tar")
        # validation data (scaled)
        df_opt = pd.DataFrame(all_ds_opt_scaled[dev_idx], columns=all_df_benign[dev_idx].columns)
        df_opt.to_pickle(f"data/fl_{dev}_valid.pickle")
        # test data (scaled)
        for dm in data_mal:
            df_dm = pd.DataFrame(dm["data"], columns=all_df_benign[dev_idx].columns)
            df_dm.to_pickle(f"data/fl_{dev}_{dm['name']}.pickle")
        # and eval labels
        for dl in data_labels:
            label_name = dl["type"].unique()[0]
            dl.to_pickle(f"data/fl_{dev}_{label_name}_labels.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process N-BaIoT dataset (FL version).")
    parser.add_argument("-b", "--basepath", type=lambda p: Path(p).absolute(), default=Path("archive.ics.uci.edu/ml/machine-learning-databases/00442/"),
                        help="N-BaIoT data directory.")
    parser.add_argument("-r", "--rounds", type=int, required=False, default=10,
                        help="Number of Federated Learning rounds.")
    parser.add_argument("-e", "--localepochs", type=int, required=False, default=2,
                        help="Number of local training epochs.")
    parser.add_argument("--lr", type=float, required=False, default=0.012,
                        help="Learning rate (same for all clients)")
    parser.add_argument("-d", "--devices", type=str, nargs="+", required=True,
                        help="List of space separated device names (each device is a client in FL).")

    args = parser.parse_args()

    print(f"Using {len(args.devices)} devices.")

    for dev in args.devices:
        if not (args.basepath / dev).is_dir():
            sys.exit(f"{args.basepath / dev} is not a directory.")

    main(args.basepath, args.devices, args.rounds, args.localepochs, args.lr)
