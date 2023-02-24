from pathlib import Path
import pandas as pd


# INFILE = "iotsim-combined-cycle-1_nmapcoapampli_labels.pickle"
# DEV = "dev"
# OUTFILE = f"{INFILE.split('.pickle')[0]}_re.pickle"

P = Path(".")
for INFILE in P.glob("*labels.pickle"):
    DEV = INFILE.name.split("_")[1]
    OUTFILE = f"{INFILE.name.split('.pickle')[0]}_re.pickle"
    print("--> ", INFILE)
    print("device: ", DEV)

    # data: pd.DataFrame
    data = pd.read_pickle(INFILE)
    print(data["type"].unique())
    data["type"] = DEV + "_" + data["type"]
    print(data["type"].unique())
    data.to_pickle(OUTFILE)
    print("<-- ", OUTFILE)
    print("\n\n")






