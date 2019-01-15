# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import re
import os

DATA_DIR = "../data"
IDLABELS_FILE = os.path.join(DATA_DIR, "docsim-idlabels.tsv")
TEXTS_FILE = os.path.join(DATA_DIR, "docsim-texts.tsv")

VOCAB_SIZE = 40000
LABELS = {"similar": 1, "different": 0}

ng_data = fetch_20newsgroups(subset="all", 
                             data_home=DATA_DIR,
                             shuffle=True,
                             random_state=42)
num_docs = len(ng_data.data)
print("#-docs in dataset:", num_docs)

cvec = CountVectorizer(max_features=VOCAB_SIZE)
tfidf = TfidfTransformer()

ids = np.arange(num_docs)
X = tfidf.fit_transform(cvec.fit_transform(ng_data.data))
y = np.array(ng_data.target)
print("after vectorization:", X.shape, y.shape)

Xtrain, Xtest, ytrain, ytest, idtrain, idtest = train_test_split(
        X, y, ids, test_size=0.1)
print("after split:", idtrain.shape, Xtrain.shape, ytrain.shape, 
      idtest.shape, Xtest.shape, ytest.shape)

V = Xtest.todense()
S = np.dot(V, V.T) / np.power(np.linalg.norm(V, 2), 2)
print("S:", S.shape)

top_threshold = np.percentile(S, 95)
bot_threshold = np.percentile(S, 5)

top_xs, top_ys = np.where(S >= top_threshold)
bot_xs, bot_ys = np.where(S <= bot_threshold)
print("#-positive labels:", len(top_xs), ", #-negative labels:", len(bot_xs))

doc_ids = set()
fidl = open(IDLABELS_FILE, "wb")
num_pos = 0
for top_x, top_y in zip(top_xs, top_ys):
    if np.random.uniform(0.0, 1.0, 1) > 0.2:
        continue
    if num_pos % 1000 == 0:
        print("{:d} pairs written, pos ({:d}), neg(0)".format(num_pos, num_pos))
    label = LABELS["similar"]
    x = int(top_x)
    y = int(top_y)
    if x == y:
        continue
    doc_ids.add(x)
    doc_ids.add(y)
    fidl.write("{:d}\t{:d}\t{:d}\n".format(label, x, y))
    num_pos += 1
num_neg = 0
for bot_x, bot_y in zip(bot_xs, bot_ys):
    if np.random.uniform(0.0, 1.0, 1) > 0.2:
        continue
    if num_neg % 1000 == 0:
        print("{:d} pairs written, pos({:d}), neg({:d})".format(
                num_pos + num_neg, num_pos, num_neg))
    label = LABELS["different"]
    x = int(bot_x)
    y = int(bot_y)
    doc_ids.add(x)
    doc_ids.add(y)
    fidl.write("{:d}\t{:d}\t{:d}\n".format(label, x, y))
    num_neg += 1
    if num_neg > num_pos:
        break
fidl.close()
print("{:d} pairs written, pos({:d}), neg({:d}), COMPLETE".format(
        num_pos + num_neg, num_pos, num_neg))

ftex = open(TEXTS_FILE, "wb")
num_written = 0
for doc_id in list(doc_ids):
    if num_written % 1000 == 0:
        print("{:d} texts written".format(num_written))
    text = ng_data.data[doc_id].encode("utf8").decode("ascii", "ignore").lower()
    text = re.sub("\\s+", " ", re.sub("\n", " ", text))
    ftex.write("{:d}\t{:s}\n".format(doc_id, text))
    num_written += 1
ftex.close()
print("{:d} texts written, COMPLETE".format(num_written))

