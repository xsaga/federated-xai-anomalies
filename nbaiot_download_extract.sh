# approx 9GB
wget --recursive --no-parent https://archive.ics.uci.edu/ml/machine-learning-databases/00442/
find archive.ics.uci.edu -iname "*.rar" -exec 7z e {} -o{}.d \;
