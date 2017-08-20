# -*- coding: utf-8 -*-
from __future__ import division, print_function
import re
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "../data/results"
LABELS = {
    "c": ["train", "valid", "test"],
    "r": ["RMSE", "Pearson", "Spearman"]
}
YLABEL = {
    "c": "accuracy",
}
LABEL_TYPE = "c"

#TASK_NAME = "Document Classification"
#TASK_FILE = os.path.join(DATA_DIR, "task1.tsv")

#TASK_NAME = "Document Similarity"
#TASK_FILE = os.path.join(DATA_DIR, "task2.tsv")

TASK_NAME = "Sentence Similarity"
TASK_FILE = os.path.join(DATA_DIR, "task3.tsv")
LABEL_TYPE = "r"

legends, col1s, col2s, col3s = [], [], [], []
fin = open(TASK_FILE, "rb")
for line in fin:
    if line.startswith("#"):
        continue
    if LABEL_TYPE == "c":
        legend, col1, col2, col3, _ = re.split(r"\t+", line.strip())
    else:
        legend, col1, col2, col3 = re.split(r"\t+", line.strip())
    legends.append(legend)
    col1s.append(float(col1))
    col2s.append(float(col2))
    col3s.append(float(col3))
fin.close()

width = 0.2
plt.bar(np.arange(len(legends)), col1s, width=width, color="r", 
        label=LABELS[LABEL_TYPE][0])
plt.bar(np.arange(len(legends))+width, col2s, width=width, color="g", 
        label=LABELS[LABEL_TYPE][1])
plt.bar(np.arange(len(legends))+(2*width), col3s, width=width, color="b", 
        label=LABELS[LABEL_TYPE][2])
plt.legend(loc="best")
plt.xticks(np.arange(len(legends)), legends)
if YLABEL.has_key(LABEL_TYPE):
    plt.ylabel(YLABEL[LABEL_TYPE])
plt.title(TASK_NAME + " - Results")

plt.show()
