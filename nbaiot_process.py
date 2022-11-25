import argparse
import sys

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


def main(nbaiot_data_basepath: Path, nbaiot_device_name: str):
    DATA_BASEPATH = nbaiot_data_basepath
    DEVICE_NAME = nbaiot_device_name
    GAFGYT_BASEDIR = Path("gafgyt_attacks.rar.d")
    MIRAI_BASEDIR = Path("mirai_attacks.rar.d")
    MODEL_SERIALIZED_PATH = Path(f"{DEVICE_NAME}_nbaiot_ae.pt")
    train_lr = 0.012
    train_epochs = 20  # 800
    SHOW = True

    torch.manual_seed(0)
    np.random.seed(0)

    df_benign = read_nbaiot_data(DATA_BASEPATH / DEVICE_NAME / "benign_traffic.csv")
    ds_trn, ds_opt = split_benign_data(df_benign)

    # The paper does not discuss feature transformations
    transformer = preprocessing.MinMaxScaler().fit(ds_trn)
    ds_trn_scaled = transformer.transform(ds_trn)
    ds_opt_scaled = transformer.transform(ds_opt)

    # batch size ?
    trn_dl = DataLoader(torch.from_numpy(ds_trn_scaled), batch_size=16)

    model = Autoencoder(115)
    if MODEL_SERIALIZED_PATH.is_file():
        print("Loading trained model.")
        model.load_state_dict(torch.load(MODEL_SERIALIZED_PATH))
    else:
        print("Training model.")
        # They provide info for lr and epochs, but not for the optimizer algorithm
        opt = optim.Adam(model.parameters(), lr=train_lr, weight_decay=1e-5)
        loss_func = F.mse_loss
        fit(model, optimizer=opt, loss_function=loss_func, epochs=train_epochs, train_generator=trn_dl)
        torch.save(model.state_dict(), MODEL_SERIALIZED_PATH)

    # compute anomaly threshold
    mse_ds_opt = model.predict_samples(torch.from_numpy(ds_opt_scaled))
    threshold = compute_threshold(mse_ds_opt)
    print("Anomaly threshold = ", threshold)

    # visualize
    fig, ax = plt.subplots()
    sns.scatterplot(x=np.arange(mse_ds_opt.shape[0]), y=mse_ds_opt, linewidth=0, ax=ax)
    ax.axhline(y=threshold, c="k")
    ax.set_title("MSE AE(dataset optim)")
    ax.set_yscale("log")
    if SHOW:
        plt.show()
    else:
        fig.savefig("1_mse_dataset_optim.png", format="png")

    # load evaluation data
    data_mal = read_nbaiot_multiple_fromdir(DATA_BASEPATH / DEVICE_NAME / GAFGYT_BASEDIR)
    data_mal.extend(read_nbaiot_multiple_fromdir(DATA_BASEPATH / DEVICE_NAME / MIRAI_BASEDIR))

    for dm in data_mal:
        dm["data"] = transformer.transform(dm["data"])

    # visualize
    fig, ax = plt.subplots(nrows=int(np.ceil((len(data_mal)) / (np.ceil(np.sqrt(len(data_mal)))))),
                           ncols=int(np.ceil(np.sqrt(len(data_mal)))))
    for i, dm in enumerate(data_mal):
        sns.scatterplot(x=np.arange(dm["data"].shape[0]), y=model.predict_samples(torch.from_numpy(dm["data"])), linewidth=0, ax=ax.flatten()[i])
        ax.flatten()[i].axhline(y=threshold, c="k")
        ax.flatten()[i].set_yscale("log")
        ax.flatten()[i].set_title(dm["name"])
    if SHOW:
        plt.show()
    else:
        fig.savefig("2_mse_attacks.png", format="png")

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
    chkpt = {"state_dict": model.state_dict(),
             "model_hash": None,
             "local_epochs": None,
             "loss": None,
             "train_loss": None,
             "num_samples": None}
    torch.save(chkpt, f"data/{DEVICE_NAME}_nbaiot_ae.tar")
    # validation data (scaled)
    df_opt = pd.DataFrame(ds_opt_scaled, columns=df_benign.columns)
    df_opt.to_pickle(f"data/{DEVICE_NAME}_valid.pickle")
    # test data (scaled)
    for dm in data_mal:
        df_dm = pd.DataFrame(dm["data"], columns=df_benign.columns)
        df_dm.to_pickle(f"data/{DEVICE_NAME}_{dm['name']}.pickle")
    # and eval labels
    for dl in data_labels:
        label_name = dl["type"].unique()[0]
        dl.to_pickle(f"data/{DEVICE_NAME}_{label_name}_labels.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process N-BaIoT dataset.")
    parser.add_argument("-b", "--basepath", type=lambda p: Path(p).absolute(), default=Path("archive.ics.uci.edu/ml/machine-learning-databases/00442/"),
                        help="N-BaIoT data directory.")
    parser.add_argument("-d", "--device", type=str, required=True,
                        help="Device name.")

    args = parser.parse_args()

    if not (args.basepath / args.device).is_dir():
        sys.exit(f"{args.basepath / args.device} is not a directory.")

    main(args.basepath, args.device)
