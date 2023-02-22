
df_raw = pd.read_pickle("data/iotsim-combined-cycle-1_miraiscanload.pickle")
results = results_container["clients_test_results"][0]
timestamps = results_container["clients_test_timestamps"][0]

th = 0.00041878226

# miraiscanload

labels = np.full(results.shape[0], 100)
labels_map = {100: "100: Unknown"}
# normal
labels[(df_raw["ip_src"]=="192.168.20.10") & (df_raw["ip_dst"]=="192.168.4.1")] = 0
labels[df_raw["ip_src"] == "192.168.4.1"] = 0
labels[df_raw["ip_src"] == "192.168.4.3"] = 0
labels_map[0] = "00: normal"

labels[ df_raw["sport"]==23 ] = 10
labels[ (df_raw["ip_src"]=="192.168.20.10") & (df_raw["sport"]==23) ] = 101
labels[ df_raw["dport"]==23 ] = 11
labels[ (df_raw["ip_src"]=="192.168.20.10") & (df_raw["dport"]==23) ] = 111
labels_map[10] = "10: sport 23"
labels_map[101] = "101: iot -> sport 23"
labels_map[11] = "11: dport 23"
labels_map[111] = "111: iot -> dport 23"


labels[ df_raw["sport"]==2323 ] = 12
labels[ df_raw["dport"]==2323 ] = 13
labels_map[12] = "12: sport 2323"
labels_map[13] = "13: dport 2323"

labels[ df_raw["sport"]==80 ] = 20
labels[ df_raw["dport"]==80 ] = 21
labels_map[20] = "20: sport 80"
labels_map[21] = "21: dport 80"

labels[ df_raw["ip_protocol"]=="ICMP" ] = 30
labels_map[30] = "30: icmp when scanning"

labels[ df_raw["sport"]==53 ] = 40
labels[ df_raw["dport"]==53 ] = 41
labels_map[40] = "40: sport 53 when scanning"
labels_map[41] = "41: dport 53 when scanning"

labels[ df_raw["sport"]==48101 ] = 50
labels[ df_raw["dport"]==48101 ] = 51
labels_map[50] = "50: sport 48101 when scanning"
labels_map[51] = "51: dport 48101 when scanning"

assert set(labels_map.keys()) == set(np.concatenate(([100],np.unique(labels))))

results_df = pd.DataFrame({"ts": timestamps, "rec_err": results, "label": labels, "type":pd.Series(labels).map(labels_map)})
fig, ax = plt.subplots()
sns.scatterplot(data=results_df, x="ts", y="rec_err", hue="type", linewidth=0, ax=ax, rasterized=True, s=10, alpha=0.6)
ax.axhline(y=th, linestyle=":", c="k")
ax.set_xlabel("timestamp")
ax.set_ylabel("MSE")
ax.set_title("Results anomaly detection")
fig.tight_layout()
fig.show()

##
####
########
###########
###########
#######
####
##

df_raw = pd.read_pickle("data/iotsim-combined-cycle-1_nmapcoapampli.pickle")
results = results_container["clients_test_results"][1]
timestamps = results_container["clients_test_timestamps"][1]

labels = np.full(results.shape[0], 100)
labels_map = {100: "100: Unknown"}

# normal
labels[(df_raw["ip_src"]=="192.168.20.10") & (df_raw["ip_dst"]=="192.168.4.1")] = 0
labels[df_raw["ip_src"] == "192.168.4.1"] = 0
labels[df_raw["ip_src"] == "192.168.4.3"] = 0
labels_map[0] = "00: normal"

labels[ (df_raw["ip_dst"]=="192.168.0.200") ] = 10
labels[ (df_raw["ip_src"]=="192.168.0.200") ] = 11
labels_map[10] = "10: dst victim"
labels_map[11] = "11: src victim"

labels[ (df_raw["ip_src"]=="192.168.35.10") ] = 20
labels[ (df_raw["ip_src"]=="192.168.35.10") & (df_raw["iat"]<4e-1) ] = 201
labels[ (df_raw["ip_dst"]=="192.168.35.10") ] = 21
labels_map[20] = "20: src scanner"
labels_map[201] = "201: src scanner iat<4e-1"
labels_map[21] = "21: dst scanner"

labels[ df_raw["ip_protocol"]=="ICMP" ] = 30
labels_map[30] = "30: icmp"

assert set(labels_map.keys()) == set(np.concatenate(([100],np.unique(labels))))

results_df = pd.DataFrame({"ts": timestamps, "rec_err": results, "label": labels, "type":pd.Series(labels).map(labels_map)})
fig, ax = plt.subplots()
sns.scatterplot(data=results_df, x="ts", y="rec_err", hue="type", linewidth=0, ax=ax, rasterized=True, s=10, alpha=0.6)
ax.axhline(y=th, linestyle=":", c="k")
ax.set_xlabel("timestamp")
ax.set_ylabel("MSE")
ax.set_title("Results anomaly detection")
fig.tight_layout()
fig.show()