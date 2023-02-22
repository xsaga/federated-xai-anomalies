from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import seaborn as sns

from feature_extractor import pcap_to_dataframe, port_hierarchy_map_iot, preprocess_dataframe
from model_ae import Autoencoder


def label_by_ip(df: pd.DataFrame, rules: List[Tuple[str, bool, str, bool, int]], default: int=-1) -> np.ndarray:
    """Label each packet based on a list of rules for IP addresses.
    Keyword arguments:
    df    -- DataFrame of extracted packet features.
    rules -- List of rules. Each rule is of type Tuple[str, bool, str,
    bool, int]. The first str, bool pair refers to the source IP
    address. The second str, bool pair refers to the destination IP
    address. The bools indicate whether the corresponding IP address
    should be included or excluded. The int is the label assigned to
    the packets that match the rule. Example: ("192.168.0.2", True,
    "192.168.0.10", True, 0) == if src addr IS 192.168.0.2 and dst
    addr IS 192.168.0.10 label as 0. ("192.168.0.2", True,
    "192.168.0.10", False, 1) == if src addr IS 192.168.0.2 and dst
    addr IS NOT 192.168.0.10 label as 1. You can refer to any IP
    address using an invalid IP address string and False.
    default -- default label assigned to packets that do not match the
    rules.
    """
    labels = np.full(df.shape[0], default)
    for srcip, srcinclude, dstip, dstinclude, label in rules:
        src_cmp = np.equal if srcinclude else np.not_equal
        dst_cmp = np.equal if dstinclude else np.not_equal
        mask = np.logical_and(src_cmp(df["ip_src"], srcip), dst_cmp(df["ip_dst"], dstip))
        labels[mask] = label

    return labels


def reconstruction_error(model, loss_function, samples):
    """Apply the model prediction to a list of samples."""
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error.numpy()


GLOBAL_MODEL_PATH = "data/global_model_cluster_coap.tar"
VALID_ATTACK_DATA_PATH = "data/iotsim-combined-cycle-1_miraidos.pickle"

rules = [('192.168.20.10', True, '192.168.4.1', True, 0),
         ('192.168.4.1', True, 'xxx', False, 0),
         (('192.168.4.3', True, 'xxx', False, 0)),
         ('192.168.20.10', True, '192.168.33.10', True, 1),
         ('192.168.33.10', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.11', True, 1),
         ('192.168.33.11', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.12', True, 1),
         ('192.168.33.12', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.33.13', True, 1),
         ('192.168.33.13', True, '192.168.20.10', True, 1),
         ('192.168.20.10', True, '192.168.0.100', True, 2),
         ('192.168.0.100', True, '192.168.20.10', True, 2),
         ('192.168.20.10', True, '192.168.18.18', True, 3),
         ('192.168.18.18', True, '192.168.20.10', True, 3)]


rules_map = {0: "normal",
             1: "Mirai C&C",
             2: "Mirai bot",
             3: "DoS victim",
             10: "Others"}

model = Autoencoder(69)
ckpt = torch.load(GLOBAL_MODEL_PATH)
model.load_state_dict(ckpt["state_dict"])
loss_func = F.mse_loss

th = 0.00041878226

df_raw = pd.read_pickle(VALID_ATTACK_DATA_PATH)
labels = label_by_ip(df_raw, rules, 10)
df = preprocess_dataframe(df_raw, port_mapping=port_hierarchy_map_iot)
timestamps = df["timestamp"].values
df = df.drop(columns=["timestamp"])

results = reconstruction_error(model, loss_func, torch.from_numpy(df.to_numpy(dtype=np.float32)))

fig, ax = plt.subplots()
sns.scatterplot(x=timestamps, y=results, color="blue", linewidth=0, s=12, alpha=0.3, ax=ax, rasterized=True)
ax.axhline(y=th, linestyle=":", c="k")
plt.show()

np.unique(labels, return_counts=True)

# subsample (reduce only DoS attack packets)
# subsample only DoS attacks
ss_w = pd.Series(labels).map({0: 0.0, 1: 0.0, 3: 1.0, 10: 1.0})
df_sampled = df.sample(frac=0.98, weights=ss_w, replace=False).sort_index()
rev_ix = df_sampled.index
# select the inverted packets
sampled_ix = df.index.difference(other=rev_ix)

fig, ax = plt.subplots()
sns.scatterplot(x=timestamps, y=results, color="blue", linewidth=0, s=12, alpha=0.3, ax=ax, rasterized=True)
sns.scatterplot(x=timestamps[sampled_ix], y=results[sampled_ix], color="red", linewidth=0, s=10, alpha=0.3, ax=ax, rasterized=True)
ax.axhline(y=th, linestyle=":", c="k")
plt.show()

np.unique(labels[sampled_ix], return_counts=True)

df_raw_red = df_raw.loc[sampled_ix].reset_index(drop=True)
print(df_raw_red.shape)

df_raw_red.to_pickle('data/iotsim-combined-cycle-1_miraidos_sampled.pickle')
