"""Cluster anomalies"""

import argparse
import io
import itertools
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import joblib
import numpy as np
import pandas as pd
import torch

import hdbscan

# https://umap-learn.readthedocs.io/en/latest/performance.html
from umap import UMAP

import scipy
from scipy.spatial import ConvexHull, QhullError

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import seaborn as sns

from fkms.algorithms import kfed
from federated_clustering_validation import calinski_harabasz as fed_ch


def points_in_circle_around(x0, y0, r, steps=10):
    """Create circle of points with radius r around x0, y0."""
    phi = np.linspace(0, 2 * np.pi, steps)
    dx = r * np.cos(phi)
    dy = r * np.sin(phi)
    return np.column_stack((dx + x0, dy + y0))


def select_position(positions, rest):
    """Find an 'acceptable' position to place the text label in a plot.
    From a list of candidate positions, return the position that
    maximizes the distance to the closest point.
    """
    distances = scipy.spatial.distance.cdist(positions, rest)
    closest = distances.min(axis=1)
    selection = np.argmax(closest)
    return positions[selection]


def scatterplot_cluster(data: Union[np.ndarray, List[np.ndarray]],
                        cluster_labels: Union[np.ndarray, List[np.ndarray]], colormap, marker_map: Dict[Any, str],
                        put_legend: bool = True, convex_hull: bool = False, centers: Optional[np.ndarray] = None, text_r: float = 0.02):
    if not isinstance(data, list):
        data = [data]
    if not isinstance(cluster_labels, list):
        cluster_labels = [cluster_labels]

    data_len = len(data)
    alpha_scatter = 0.33
    alpha_hull_lines = 0.3
    alpha_text_box = 0.3

    fig, ax = plt.subplots(nrows=int(np.ceil(data_len / 2)), ncols=2 if data_len > 1 else 1)
    for i in range(data_len):
        data_i = data[i]
        labels_i = cluster_labels[i]
        if data_len > 1:
            ax_i = ax.flatten()[i]
        else:
            ax_i = ax

        # scatterplot for each color
        for lbl in np.unique(labels_i):
            ax_i.scatter(x=data_i[labels_i == lbl, 0], y=data_i[labels_i == lbl, 1],
                         color=colormap(lbl), alpha=alpha_scatter,
                         marker=marker_map[lbl], label=None)

        # create legend
        # https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
        if put_legend:
            for lbl in np.unique(labels_i):
                ax_i.scatter([], [],
                             color=colormap(lbl),
                             marker=marker_map[lbl], label=str(lbl))
            ax_i.legend(title="Cluster")

        # draw Convex Hull
        if convex_hull:
            for lbl in np.unique(labels_i):
                if lbl < 0:  # ignore outliers
                    continue
                data_i_c = data_i[labels_i == lbl]
                if data_i_c.shape[0] < 3:
                    # not enough points for convex hull
                    if data_i_c.shape[0] == 2:
                        ax_i.plot(data_i_c[:, 0], data_i_c[:, 1],
                                  linestyle="solid", color=colormap(lbl), alpha=alpha_hull_lines)

                try:
                    hull = ConvexHull(data_i_c)
                    for simplex in hull.simplices:
                        ax_i.plot(data_i_c[simplex, 0], data_i_c[simplex, 1],
                                  linestyle="solid", color=colormap(lbl), alpha=alpha_hull_lines)
                except QhullError:
                    pass

        # Show centers
        if isinstance(centers, np.ndarray):
            ax_i.scatter(centers[:, 0], centers[:, 1], color="k", marker="x")

            for lbl in np.unique(labels_i):
                if lbl < 0:
                    continue
                position_options = points_in_circle_around(centers[lbl, 0], centers[lbl, 1], text_r, 10)
                other_points = np.delete(centers, lbl, axis=0)
                text_x, text_y = select_position(position_options, other_points)
                ax_i.text(x=text_x, y=text_y, s=f"C {lbl}", bbox=dict(facecolor=colormap(lbl), edgecolor=colormap(lbl), alpha=alpha_text_box, boxstyle="round", pad=0.1), horizontalalignment="center")

        if data_len > 1:
            ax_i.set_title(f"Client #{i}")

    return fig, ax


def make_table_client_clusters(clients_names: List[str], clients_labels: List[np.ndarray], percent=True, round_decimals=2, zeros_are_nan=True) -> pd.DataFrame:
    assert len(clients_names) == len(clients_labels)
    all_uniq_labels = np.unique(np.concatenate(clients_labels))
    results = {}
    for name, labels in zip(clients_names, clients_labels):
        counts = np.array([np.count_nonzero(labels == l) for l in all_uniq_labels], dtype=np.float32)
        if zeros_are_nan:
            counts[counts == 0] = np.nan
        if percent:
            counts = np.around((counts / np.nansum(counts))*100, decimals=round_decimals)
        results[name] = counts
    return pd.DataFrame(results, index=all_uniq_labels).T


def one_vs_all_labels(original_labels: np.ndarray, selected_label, other_label=0, self_label=1) -> np.ndarray:
    ova = np.full_like(original_labels, fill_value=other_label)
    ova[original_labels == selected_label] = self_label
    return ova


def get_closest_samples_to_centers(centers: np.ndarray, labels: List[np.ndarray], samples: List[np.ndarray]) -> Dict[int, List[Optional[np.ndarray]]]:
    """For each client and cluster center, select the sample nearest to the cluster center"""
    closest_samples: Dict[int, List[Optional[np.ndarray]]] = {}

    for k in range(0, centers.shape[0]):
        k_center = centers[k, :]
        k_closest = []
        for i in range(len(labels)):
            i_labels = labels[i]
            if k in i_labels:
                ki_samples = samples[i][i_labels == k]
                dist = scipy.spatial.distance.cdist(ki_samples, k_center.reshape(1, -1), metric="euclidean")
                k_closest.append(ki_samples[np.argmin(dist), :])
            else:
                k_closest.append(None)
        closest_samples[k] = k_closest

    return closest_samples


def heatmap_shap_cluster_client(client_idx: int, shap_closest_samples_center: Dict[int, List[Optional[np.ndarray]]], feature_names: List[str]):
    """Plot heatmap of shap values / cluster / client"""
    fig, ax = plt.subplots()

    shap_clus_values = []
    cluster_names = []

    for c, v in shap_closest_samples_center.items():
        if v[client_idx] is None:
            continue
        shap_clus_values.append(v[client_idx])
        cluster_names.append(f"C{c}")

    shap_clus_values = np.array(shap_clus_values)
    sns.heatmap(shap_clus_values, cmap="bwr", center=0, square=True, xticklabels=feature_names, yticklabels=cluster_names, cbar_kws={"shrink": 0.33})

    return fig, ax


def save_summary_as_excel(fname: str,
                          shap_cluster_centers: Dict[int, Optional[np.ndarray]],
                          cluster_labels: np.ndarray,
                          samples: pd.DataFrame,
                          samples_raw: Optional[pd.DataFrame] = None,
                          time_anom: Optional[np.ndarray] = None
                          ) -> None:
    """ Save results to an excel file.
    - 1 Excel sheet for each cluster label. Only if there are samples for said cluster.
    - Each sheet includes 2 (or 3 if samples_raw not None) tables and an image if time_anom is not None
    """
    img_io_objs = []  # List[io.BytesIO], only close after pd.ExcelWriter is closed.
    with pd.ExcelWriter(fname) as writer:
        for k, shap_center in shap_cluster_centers.items():
            if shap_center is None:
                continue

            sheet_name = f"Cluster {k}"
            row_cnt = 0

            print(f"--- k={k} ---")
            ova_labels = one_vs_all_labels(cluster_labels, k, other_label=0, self_label=1)

            # shap
            shap_series = pd.Series(shap_center, index=samples.columns, name="SHAP").sort_values(key=np.abs, ascending=False)
            shap_series.to_excel(writer, startrow=row_cnt, startcol=0, sheet_name=sheet_name)
            row_cnt += (shap_series.shape[0] + 1 + 2)

            # images, if available:
            if time_anom is not None:
                fig, ax = plt.subplots()
                ax.scatter(time_anom[ova_labels == 0, 0], time_anom[ova_labels == 0, 1], alpha=0.2, color="#619CFF", label="rest")
                ax.scatter(time_anom[ova_labels == 1, 0], time_anom[ova_labels == 1, 1], alpha=0.2, color="#F8766D", label="self")
                ax.legend()
                ax.set_title(f"Samples from cluster {k} ({ova_labels.sum()})")
                fig.tight_layout()
                img_io = io.BytesIO()
                fig.savefig(img_io, format="png")
                writer.sheets[sheet_name].insert_image(1, 3, sheet_name, {"image_data": img_io})
                plt.close(fig)
                img_io_objs.append(img_io)

            # samples
            samples_k = samples.loc[ova_labels == 1].describe(include="all")
            samples_k.to_excel(writer, startrow=row_cnt, startcol=0, sheet_name=sheet_name)
            row_cnt += (samples_k.shape[0] + 1 + 2)

            # samples raw if available
            if samples_raw is not None:
                samples_raw_k = samples_raw.loc[ova_labels == 1].describe(include="all")
                samples_raw_k.to_excel(writer, startrow=row_cnt, startcol=0, sheet_name=sheet_name)

    for img_io in img_io_objs:
        img_io.close()


def reconstruction_error(model, loss_function, samples):
    """Apply the model prediction to a list of samples."""
    with torch.no_grad():
        model.eval()
        predictions = model(samples)
        rec_error = torch.mean(loss_function(samples, predictions, reduction="none"), dim=1)
    return rec_error.numpy()


def model_predict(X, model, loss_func):
    return reconstruction_error(model, loss_func, torch.from_numpy(X))


parser = argparse.ArgumentParser(description="Cluster anomalies in a FL setting.")
parser.add_argument("--umap", action="store_true",
                    help="Show 2D dimensionality reduction with UMAP.")
parser.add_argument("-i", "--input", type=lambda p: Path(p).absolute(), required=True,
                    help="SHAP value results.")
parser.add_argument("-l", "--labels", type=lambda p: Path(p).absolute(), nargs="+", required=True,
                    help="Manual labelling data.")

args = parser.parse_args()

OUTPUT_DIR = Path(f"results_{datetime.now().strftime('%Y_%m_%dT%H_%M_%S')}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# general matplotlib config
rcParams["font.family"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["axes.labelsize"] = 8
rcParams["legend.fontsize"] = 8
# rcParams["lines.linewidth"] = 0.75
# rcParams["lines.markersize"] = 2
plot_width = 3.487  # in
plot_height = 2.325  # 2.155


#############
# Load data #
#############

with open(args.input, "rb") as f:
    results_container = pickle.load(f)

assert len(results_container["clients_test_dfs"]) == len(args.labels)
num_clients = len(results_container["clients_test_dfs"])

# load manual labels
clients_labels = list(map(pd.read_pickle, args.labels))

# get labels of anomalous samples
clients_anomalies_labels = []
for i in range(num_clients):
    idx = results_container["clients_anomalies"][i].index
    lab = clients_labels[i]
    clients_anomalies_labels.append(lab.iloc[idx])

# clients_anomalies_labels[1]["type"][clients_anomalies_labels[1]["type"] == "Victim"] = "Victim CoAP"

clients_anomalies_shap = results_container["clients_anomalies_shap"]

# centralized shap values
centralized_anomalies_shap = np.concatenate(clients_anomalies_shap)
centralized_anomalies_client_label = np.concatenate([np.full(clients_anomalies_shap[i].shape[0], i) for i in range(num_clients)])
centralized_anomalies_labels = np.concatenate([clients_anomalies_labels[i]["type"] for i in range(num_clients)])

# preprocess shap values
clients_anomalies_shap_proc = list(map(preprocessing.normalize, clients_anomalies_shap))
centralized_anomalies_shap_proc = preprocessing.normalize(centralized_anomalies_shap)

#######################
# Local visualization #
#######################

# visualize: 2d projection shap values for each client, colorized by local labels
fig, ax = plt.subplots(nrows=int(np.ceil(num_clients / 2)), ncols=2)
for i in range(num_clients):
    c_shap_2d = PCA(n_components=2).fit_transform(clients_anomalies_shap_proc[i])
    sns.scatterplot(x=c_shap_2d[:, 0], y=c_shap_2d[:, 1], hue=clients_anomalies_labels[i]["type"], palette="deep", linewidth=0, alpha=0.33, ax=ax.flatten()[i], rasterized=True)
    ax.flatten()[i].set_title(f"Client #{i}")
fig.tight_layout()
plt.show()

if args.umap:
    fig, ax = plt.subplots(nrows=int(np.ceil(num_clients / 2)), ncols=2)
    for i in range(num_clients):
        c_shap_2d = UMAP().fit_transform(clients_anomalies_shap_proc[i])
        sns.scatterplot(x=c_shap_2d[:, 0], y=c_shap_2d[:, 1], hue=clients_anomalies_labels[i]["type"], palette="deep", linewidth=0, alpha=0.33, ax=ax.flatten()[i], rasterized=True)
        ax.flatten()[i].set_title(f"Client #{i}")
    fig.tight_layout()
    plt.show()

#############################
# Centralized visualization #
#############################

centralized_pca_explained_var = 0.99
centralized_pca = PCA(n_components=centralized_pca_explained_var).fit(centralized_anomalies_shap_proc)
print(f"Centralized PCA: {centralized_pca.n_components_} components to explain {np.sum(centralized_pca.explained_variance_ratio_)} var")

centralized_shap_2d = centralized_pca.transform(centralized_anomalies_shap_proc)[:, :2]

# visualize: centralized 2d projection shap values, colorized by client
fig, ax = plt.subplots()
sns.scatterplot(x=centralized_shap_2d[:, 0], y=centralized_shap_2d[:, 1], hue=centralized_anomalies_client_label, palette="deep", linewidth=0, alpha=0.33, ax=ax)
plt.show()

# visualize: centralized 2d projection shap values, colorized by manual label
fig, ax = plt.subplots()
sns.scatterplot(x=centralized_shap_2d[:, 0], y=centralized_shap_2d[:, 1], hue=centralized_anomalies_labels, palette="deep",
                style=centralized_anomalies_client_label, linewidth=0, alpha=0.33, ax=ax)
plt.show()

# visualize: final
_cent_pca_df = pd.DataFrame({"principal component 1": centralized_shap_2d[:, 0],
                             "principal component 2": centralized_shap_2d[:, 1],
                             "Label": centralized_anomalies_labels,
                             "Client": centralized_anomalies_client_label})
fig, ax = plt.subplots()
sns.scatterplot(data=_cent_pca_df, x="principal component 1", y="principal component 2",
                hue="Label", palette="deep", edgecolor="none", alpha=0.1, s=1, rasterized=True, ax=ax)  # , style="Client", s=5
sns.move_legend(ax, "upper left", bbox_to_anchor=(0.98, 1), fancybox=False, frameon=False, borderpad=0.0, handletextpad=0.0, fontsize=6)
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.savefig(f"{args.input.stem}_centralized_shap_pca_2d.pdf", format="pdf")
# plt.show()

# pca explained variance
fig, ax = plt.subplots()
ax.plot(np.arange(centralized_pca.n_components_) + 1, np.cumsum(centralized_pca.explained_variance_ratio_), marker="o")
plt.show()

if args.umap:
    centralized_umap = UMAP(random_state=42)  # UMAP(random_state=42) Setting a random state disables non-reproducible multi-threading and can take longer!
    centralized_shap_2d_umap = centralized_umap.fit_transform(centralized_anomalies_shap_proc)

    # save (for large sets, UMAP can take hours)
    np.save(args.input.stem + "_2d_umap.npy", centralized_shap_2d_umap)
    joblib.dump(centralized_umap, args.input.stem + "_UMAP_obj.gz")
    # centralized_shap_2d_umap = np.load(args.input.stem + "_2d_umap.npy")
    # centralized_umap = joblib.load(args.input.stem + "_UMAP_obj.gz")


    fig, ax = plt.subplots()
    sns.scatterplot(x=centralized_shap_2d_umap[:, 0], y=centralized_shap_2d_umap[:, 1], hue=centralized_anomalies_client_label, palette="deep", linewidth=0, alpha=0.33, ax=ax)
    plt.show()

    fig, ax = plt.subplots()
    sns.scatterplot(x=centralized_shap_2d_umap[:, 0], y=centralized_shap_2d_umap[:, 1], hue=centralized_anomalies_labels, palette="deep",
                    style=centralized_anomalies_client_label, linewidth=0, alpha=0.33, ax=ax)
    plt.show()

    # final
    _cent_umap_df = pd.DataFrame({"UMAP embedding 1": centralized_shap_2d_umap[:, 0],
                                  "UMAP embedding 2": centralized_shap_2d_umap[:, 1],
                                  "Label": centralized_anomalies_labels,
                                  "Client": centralized_anomalies_client_label})
    fig, ax = plt.subplots()
    sns.scatterplot(data=_cent_umap_df, x="UMAP embedding 1", y="UMAP embedding 2",
                    hue="Label", palette="deep", edgecolor="none", alpha=0.1, s=1, rasterized=True, ax=ax)  # , style="Client", s=5
    sns.move_legend(ax, "upper left", bbox_to_anchor=(0.98, 1), fancybox=False, frameon=False, borderpad=0.0, handletextpad=0.0, fontsize=6)
    fig.set_size_inches(plot_width, plot_height)
    fig.tight_layout()
    fig.savefig(f"{args.input.stem}_centralized_shap_umap_2d.pdf", format="pdf")
    # plt.show()



##########################
# centralized clustering #
##########################

# Centralized (H)DBSCAN
# the HDBSCAN library also provides DBSCAN clustering from HDBSCAN, more memory efficient than scikit-learns DBSCAN
# for a single trained HDBSCAN, we can extract any DBSCAN clustering results (any epsilon) without retraining
# https://github.com/scikit-learn-contrib/hdbscan/blob/2179c24a31742aab459c75ac4823acad2dca38cf/docs/dbscan_from_hdbscan.rst
# https://hdbscan.readthedocs.io/en/latest/parameter_selection.html (cluster_selection_epsilon)
# https://hdbscan.readthedocs.io/en/latest/how_to_use_epsilon.html
# centralized_clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=1)
centralized_clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=1, cluster_selection_epsilon=0.05, cluster_selection_method="eom")  # 100, 1, 0.01
centralized_clusterer.fit(centralized_pca.transform(centralized_anomalies_shap_proc))
centralized_unsupervised_labels = centralized_clusterer.labels_
print(np.unique(centralized_unsupervised_labels, return_counts=True))

cmap = cm.get_cmap("hsv", np.unique(centralized_unsupervised_labels).shape[0])
marker_map = dict(zip(np.unique(centralized_unsupervised_labels),
                      itertools.cycle("ov^<>spP*HXD")))
fig, ax = scatterplot_cluster(centralized_shap_2d, centralized_unsupervised_labels, cmap, marker_map, True, True,
                              np.array([centralized_clusterer.weighted_cluster_centroid(i) for i in range(centralized_unsupervised_labels.max() + 1)])[:, :2],
                              0.03)
plt.show()

# plot t/anom
fig, ax = scatterplot_cluster([np.column_stack((t[a.index], r[a.index])) for t, r, a in zip(results_container["clients_test_timestamps"], results_container["clients_test_results"], results_container["clients_anomalies"])],  # List[np.ndarray], for each ndarray column 0 = timestamp, column 1 = reconstruction error, only select samples considered anomalies
                              np.split(centralized_unsupervised_labels, np.cumsum(list(map(lambda x: x.shape[0], results_container["clients_anomalies"])))),  # split the array into num_clients sub-arrays with the correct size (filtered for anomalies) for each sub-array
                              cmap, marker_map, True, False, False)
fig.tight_layout()
plt.show()


# plot t/anom by each label
for lbl in np.unique(centralized_unsupervised_labels):
    print(f"--> lbl = {lbl}")
    cmap = cm.get_cmap("jet", 2)
    marker_map = {0: "o", 1: "^"}
    fig, ax = scatterplot_cluster([np.column_stack((t[a.index], r[a.index])) for t, r, a in zip(results_container["clients_test_timestamps"], results_container["clients_test_results"], results_container["clients_anomalies"])],
                                  np.split(one_vs_all_labels(centralized_unsupervised_labels, lbl), np.cumsum(list(map(lambda x: x.shape[0], results_container["clients_anomalies"])))),
                                  cmap, marker_map, True, False, False)
    fig.tight_layout()
    plt.show()

########################
# Federated Clustering #
########################

# estimate local k for each client (= k'_i)
clients_k_p = []
for i in range(num_clients):
    print(f"Client #{i}", end="", flush=True)
    c_pca = PCA(n_components=0.99).fit(clients_anomalies_shap_proc[i])
    print(".", end="", flush=True)
    clus = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=1, cluster_selection_epsilon=0.05, cluster_selection_method="eom", allow_single_cluster=True)
    clus.fit(c_pca.transform(clients_anomalies_shap_proc[i]))
    print(": ", end="", flush=True)
    k_p = clus.labels_.max() + 1  # ignore outliers
    print(f"(k' = {k_p})\nlocal HDBSCAN labels: {np.unique(clus.labels_, return_counts=True)}")
    clients_k_p.append(k_p)

k_p = max(clients_k_p)
print(f"Final k' = {k_p}")

np.random.seed(10)
random.seed(10)
k_fed_repetitions = 30
k_fed_results: Dict[int, List[Dict[str, Union[np.ndarray, List[np.ndarray]]]]] = {}
#                   ^ k-fed global k
#                        ^ list of length k_fed_repetitions
#                                  ^ key = "centers" value = np.ndarray of shape (k, data_dimensions)
#                                  ^ key = "labels" value = list of length num_clients where each item is np.ndarray of shape (num_samples_client,)

for k in range(k_p, k_p * num_clients):
    rep_k = []
    for r in range(k_fed_repetitions):
        print(f"K-FED, k={k} (repetition {r})")
        # federated clustering
        local_estimates, federated_centers, global_counts = kfed(clients_anomalies_shap_proc, k_p, k, round_values=False)
        # get all labels
        k_fed_labels = []
        for i in range(num_clients):
            local_km = KMeans(n_clusters=k, init=federated_centers, n_init=1).fit(federated_centers)
            local_labels = local_km.predict(clients_anomalies_shap_proc[i])
            k_fed_labels.append(local_labels)

        rep_k.append({"centers": federated_centers, "labels": k_fed_labels})
    k_fed_results[k] = rep_k

joblib.dump(k_fed_results, args.input.stem + "_k_fed_results.gz")
# k_fed_results = joblib.load(args.input.stem + "_k_fed_results.gz")

# compare clustering quality with centralized
ari_scores = []
ami_scores = []
vme_scores = []
for k in k_fed_results.keys():
    ari_s = []
    ami_s = []
    vme_s = []
    for r in range(k_fed_repetitions):
        kf_lbl = np.concatenate(k_fed_results[k][r]["labels"])
        ari_s.append(adjusted_rand_score(kf_lbl, centralized_unsupervised_labels))
        ami_s.append(adjusted_mutual_info_score(kf_lbl, centralized_unsupervised_labels))
        vme_s.append(v_measure_score(kf_lbl, centralized_unsupervised_labels))
    ari_scores.append(ari_s)
    ami_scores.append(ami_s)
    vme_scores.append(vme_s)

ari_df = pd.DataFrame(np.array(ari_scores).T, columns=list(k_fed_results.keys())).assign(metric="ARI")
ami_df = pd.DataFrame(np.array(ami_scores).T, columns=list(k_fed_results.keys())).assign(metric="AMI")
vme_df = pd.DataFrame(np.array(vme_scores).T, columns=list(k_fed_results.keys())).assign(metric="VME")
metrics_df = pd.concat([ari_df, ami_df, vme_df])
metrics_df = pd.melt(metrics_df, id_vars=["metric"], var_name="num_clusters", value_name="score")
fig, ax = plt.subplots()
sns.boxplot(data=metrics_df, x="num_clusters", y="score", hue="metric", ax=ax)
plt.show()


# boxplot ARI centralized
fig, ax = plt.subplots()
# lll = [(l if l%4==0 else "") for l in list(ari_df.drop(columns=["metric"]).columns)]  # labels=lll
ax.boxplot(ari_df.drop(columns=["metric"]).to_numpy(),
           labels=list(ari_df.drop(columns=["metric"]).columns), positions=list(ari_df.drop(columns=["metric"]).columns),
           capprops=dict(color="black", linewidth=1))
ax.hlines(y=1.0, xmin=k_p, xmax=(k_p * num_clients)-1, colors="silver", linestyles="dotted")
ax.set_xlabel("Number of global clusters")
ax.set_ylabel("Adjusted Rand score")
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.savefig(f"{args.input.stem}_ari_fl_centralized.pdf", format="pdf")
# plt.show()


# unsupervised federated clustering quality metrics
k_fed_cluster_quality = {}
for k, v in k_fed_results.items():
    k_fed_cluster_quality[k] = [fed_ch.calinski_harabasz_score_federated(clients_anomalies_shap_proc, x["labels"], x["centers"]) for x in v]
k_fed_cluster_quality = pd.DataFrame(k_fed_cluster_quality)

# boxplot federated unsupervised
fig, ax = plt.subplots()
ax.boxplot(k_fed_cluster_quality.to_numpy(),
           labels=k_fed_cluster_quality.columns, positions=k_fed_cluster_quality.columns,
           capprops=dict(color="black", linewidth=1))
ax.set_xlabel("Number of global clusters")
ax.set_ylabel("Calinski-Harabasz score")
fig.set_size_inches(plot_width, plot_height)
fig.tight_layout()
fig.savefig(f"{args.input.stem}_CHscore_fl.pdf", format="pdf")
# plt.show()


gK = 22  #24
# gKix = (np.array(list(k_fed_results.keys())) == gK).nonzero()[0][0]
# k_fed_results_best = k_fed_results[gK][np.argmax(ari_scores[gKix])]
k_fed_results_best = k_fed_results[gK][np.argmax(k_fed_cluster_quality[gK])]

# visualize k-fed in the centralized setting: label the centralized data using the centers from the federated KMeans method.
# fake KMeans
fake_km = KMeans(n_clusters=gK, init=k_fed_results_best["centers"], n_init=1).fit(k_fed_results_best["centers"])
assert np.all(np.isclose(k_fed_results_best["centers"], fake_km.cluster_centers_))
fake_labels = np.concatenate(k_fed_results_best["labels"])
cmap = cm.get_cmap("hsv", gK)
marker_map = dict(zip(range(gK),
                      itertools.cycle("ov^<>spP*HXD")))
fig, ax = scatterplot_cluster(centralized_shap_2d, fake_labels, cmap, marker_map, True, True,
                              centralized_pca.transform(fake_km.cluster_centers_)[:, :2], 0.03)
plt.show()

# plot en t/anom
fig, ax = scatterplot_cluster([np.column_stack((t[a.index], r[a.index])) for t, r, a in zip(results_container["clients_test_timestamps"], results_container["clients_test_results"], results_container["clients_anomalies"])],  # List[np.ndarray], for each ndarray column 0 = timestamp, column 1 = reconstruction error, only select samples considered anomalies
                              k_fed_results_best["labels"],
                              cmap, marker_map, True, False, False)
fig.tight_layout()
plt.show()

# table
table_client_cluster = make_table_client_clusters([p.stem for p in args.labels],
                                                  k_fed_results_best["labels"])
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.float_format", "{:.2f}".format):
    print(table_client_cluster)
table_client_cluster.to_latex(buf=f"{args.input.stem}_fed_kmeans_cluster_table.tex",
                              na_rep="-", float_format="%.2f")

# get the closest SHAP sample for each cluster center
closest_samples = get_closest_samples_to_centers(k_fed_results_best["centers"], k_fed_results_best["labels"], clients_anomalies_shap_proc)

# fig, ax = plt.subplots()
# ax.scatter(x=centralized_shap_2d[:, 0], y=centralized_shap_2d[:, 1], alpha=0.33)

# for k in range(0, gK):
#     if closest_samples[k][0] is not None:
#         ax.scatter(x=centralized_pca.transform(closest_samples[k][0].reshape(1, -1))[:, 0], y=centralized_pca.transform(closest_samples[k][0].reshape(1, -1))[:, 1], s=10, color="r", marker="o")
#     if closest_samples[k][1] is not None:
#         ax.scatter(x=centralized_pca.transform(closest_samples[k][1].reshape(1, -1))[:, 0], y=centralized_pca.transform(closest_samples[k][1].reshape(1, -1))[:, 1], s=10, color="b", marker="o")
# ax.scatter(x=centralized_pca.transform(k_fed_results_best["centers"])[:, 0], y=centralized_pca.transform(k_fed_results_best["centers"])[:, 1], color="k", marker="x")
# plt.show()

# get original data, not shap.
clients_dfs = results_container["clients_test_dfs"]
clients_dfs_raw = list(map(pd.read_pickle, results_container["clients_test_data_path"]))
feat_names = clients_dfs[0].columns
feat_names_raw = clients_dfs_raw[0].columns
# aligned with anomalous index
clients_dfs_anom = [d.loc[a.index] for d, a in zip(results_container["clients_test_dfs"], results_container["clients_anomalies"])]
clients_dfs_raw_anom = [d.loc[a.index] for d, a in zip(list(map(pd.read_pickle, results_container["clients_test_data_path"])), results_container["clients_anomalies"])]
clients_recerror_anom = [r[a.index] for r, a in zip(results_container["clients_test_results"], results_container["clients_anomalies"])]
clients_timestamp_anom = [t[a.index] for t, a in zip(results_container["clients_test_timestamps"], results_container["clients_anomalies"])]


# Results, SHAP and sample summary for each cluster
for i in range(num_clients):
    save_summary_as_excel(OUTPUT_DIR / f"client-{i}.xlsx",
                          {k: v[i] for k, v in closest_samples.items()},
                          k_fed_results_best["labels"][i],
                          clients_dfs_anom[i],
                          clients_dfs_raw_anom[i],
                          np.column_stack((clients_timestamp_anom[i], clients_recerror_anom[i])))


# Plot heatmap of shap values / cluster / client
for i in range(num_clients):
    print(f"Client #{i}")
    fig, ax = heatmap_shap_cluster_client(i, closest_samples, feat_names.str.replace("_", " "))
    fig.set_size_inches(plot_width*3, plot_height*2.5)
    fig.tight_layout()
    # plt.show()
    fig.savefig(f"{args.input.stem}_heatmap_shap_clus_client_{i}.pdf", format="pdf")
    plt.close(fig)

# Results, Trees
# Include here the normal samples (label normal samples as gK)
clients_fed_labels = [np.full(cdf.shape[0], gK) for cdf in clients_dfs]
for x, a, l in zip(clients_fed_labels, results_container["clients_anomalies"], k_fed_results_best["labels"]):
    np.put(x, a.index, l)

# multi class anomalies
for i in range(num_clients):
    print(f"--- Client #{i} ---")
    clf = tree.DecisionTreeClassifier().fit(clients_dfs[i], clients_fed_labels[i])
    print(clf.score(clients_dfs[i], clients_fed_labels[i]))
    print(tree.export_text(clf, feature_names=list(feat_names), decimals=3, show_weights=True))
    tree.plot_tree(clf, feature_names=feat_names, class_names=[f"Cluster {i}" for i in np.unique(clients_fed_labels[i])], filled=True)
    plt.show()

# one vs all
with open(OUTPUT_DIR / "ova_tree_results.html", "w") as tree_outfile:
    tree_outfile.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>One vs all decision tree classifiers</h1>\n")
    tree_plot_outdir = OUTPUT_DIR / "trees"
    tree_plot_outdir.mkdir(parents=True, exist_ok=True)
    for i in range(num_clients):
        print(f"--- Client #{i} ---")
        tree_outfile.write(f"<h2>Client #{i}</h2>\n")
        for k in np.unique(clients_fed_labels[i]):
            if k == gK:
                continue
            print(f"\tk = {k}")
            ova_i_k = one_vs_all_labels(clients_fed_labels[i], k, other_label=-1, self_label=1)
            tree_outfile.write(f"<h3>Client #{i}, Cluster k = {k} ({np.count_nonzero(ova_i_k==1)})</h3>\n")
            clf = tree.DecisionTreeClassifier(max_depth=4).fit(clients_dfs[i], ova_i_k)
            clf_txt = tree.export_text(clf, feature_names=list(feat_names), decimals=3, show_weights=True)
            print("\t", clf.score(clients_dfs[i], ova_i_k))
            print(clf_txt)
            fig, ax = plt.subplots()
            tree.plot_tree(clf, feature_names=feat_names, class_names=["other", "self"], filled=True, ax=ax)
            # Or, savefig to StringIO and embedd it directly in the html.
            tree_plot_outfile = tree_plot_outdir / f"ova_tree_c{i}_k{k}.svg"
            fig.savefig(tree_plot_outfile, format="svg")
            plt.close(fig)
            tree_outfile.write(f"<img src='{tree_plot_outfile.relative_to(OUTPUT_DIR)}' alt='tree for client {i} cluster {k}'>\n")
            tree_outfile.write(f"<pre>\n{clf_txt}\n</pre>\n")
    tree_outfile.write("</body>\n</html>\n")

# Random forest with limited number of trees and restricted depth
# worse than a regular decision tree classifier
with open(OUTPUT_DIR / "ova_forest_results.html", "w") as forest_outfile:
    forest_outfile.write("<!DOCTYPE html>\n<html>\n<body>\n<h1>One vs all random forest classifiers</h1>\n")
    forest_plot_outdir = OUTPUT_DIR / "forest"
    forest_plot_outdir.mkdir(parents=True, exist_ok=True)
    for i in range(num_clients):
        print(f"--- Client #{i} ---")
        forest_outfile.write(f"<h2>Client #{i}</h2>\n")
        for k in np.unique(clients_fed_labels[i]):
            if k == gK:
                continue
            print(f"\tk = {k}")
            ova_i_k = one_vs_all_labels(clients_fed_labels[i], k, other_label=-1, self_label=1)
            forest_outfile.write(f"<h3>Client #{i}, Cluster k = {k} ({np.count_nonzero(ova_i_k==1)})</h3>\n")
            clf = RandomForestClassifier(n_estimators=5, max_depth=2, bootstrap=True).fit(clients_dfs[i], ova_i_k)
            print("\t", clf.score(clients_dfs[i], ova_i_k))
            for e, estimator in enumerate(clf.estimators_):
                forest_outfile.write(f"<h4>Client #{i}, Cluster k = {k} ({np.count_nonzero(ova_i_k==1)}), estimator {e}</h4>\n")
                fig, ax = plt.subplots()
                tree.plot_tree(estimator, feature_names=feat_names, class_names=["other", "self"], filled=True, ax=ax)
                forest_plot_outfile = forest_plot_outdir / f"ova_forest_c{i}_k{k}_e{e}.svg"
                fig.savefig(forest_plot_outfile, format="svg")
                plt.close(fig)
                forest_outfile.write(f"<img src='{forest_plot_outfile.relative_to(OUTPUT_DIR)}' alt='forest for client {i} cluster {k} estimator {e}'>\n")
    forest_outfile.write("</body>\n</html>\n")


# ~~~ ignore ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# clf = xgb.XGBClassifier(nthread=1, max_depth=3).fit(clients_dfs[i_], ova_labels)
# print(pd.Series(clf.feature_importances_, index=clients_dfs[i_].columns).sort_values(ascending=False).head(10))
# xgb.plot_tree(clf); plt.show()

# query_labels = clients_dfs[i_].eval("sport_httpPorts>=1")
# manual_labels = np.full(query_labels.shape, 0)
# manual_labels[query_labels] = 1
# print(accuracy_score(manual_labels, ova_labels))
# neq = np.not_equal(manual_labels, ova_labels)
# manual_labels[neq] = 2
# sns.scatterplot(x=clients_timestamp[i_], y=clients_recerror[i_], hue=manual_labels.astype(str), linewidth=0, s=8, alpha=0.6); plt.show()
