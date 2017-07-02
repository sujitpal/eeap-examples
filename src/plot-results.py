# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = "../data/results"

#TASK_NAME = "Document Classification"
#TASK_FILE = os.path.join(DATA_DIR, "task1.tsv")

#TASK_NAME = "Document Similarity"
#TASK_FILE = os.path.join(DATA_DIR, "task2.tsv")

TASK_NAME = "Sentence Similarity"
TASK_FILE = os.path.join(DATA_DIR, "task3.tsv")

legends, test_accs, train_accs, val_accs = [], [], [], []
fin = open(TASK_FILE, "rb")
for line in fin:
    if line.startswith("#"):
        continue
    legend, test_acc, train_acc, val_acc, _ = line.strip().split("\t")
    legends.append(legend)
    test_accs.append(float(test_acc))
    train_accs.append(float(train_acc))
    val_accs.append(float(val_acc))
fin.close()

width = 0.2
plt.bar(np.arange(len(legends)), train_accs, width=width, color="r", label="train")
plt.bar(np.arange(len(legends))+width, val_accs, width=width, color="g", label="valid")
plt.bar(np.arange(len(legends))+(2*width), test_accs, width=width, color="b", label="test")
plt.legend(loc="best")
plt.xticks(np.arange(len(legends)), legends)
plt.ylabel("accuracy")
plt.title(TASK_NAME + " - Results")

plt.show()
