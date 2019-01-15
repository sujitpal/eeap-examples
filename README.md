# eeap-examples

## Table of Contents

* [Introduction](#introduction)
* [Data](#data)
* [Examples](#examples)
  * [Document Classification Task](#document-classification-task)
      * [02-ng-clf-ep.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/02-ng-clf-ep.ipynb)
      * [03-ng-clf-eep.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/03-ng-clf-eep.ipynb)
      * [04a-ng-clf-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/04a-ng-clf-eeap.ipynb)
      * [04b-ng-clf-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/04b-ng-clf-eeap.ipynb)
      * [04c-ng-clf-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/04c-ng-clf-eeap.ipynb)
  * [Document Similarity Task](#document-similarity-task)
      * [05a-ng-sim-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/05a-ng-sim-eep.ipynb)
      * [05b-ng-sim-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/05b-ng-sim-eeap.ipynb)
      * [05c-ng-sim-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/05c-ng-sim-eeap.ipynb)
  * [Sentence Similarity Task](#sentence-similarity-task)
      * [07a-sts-sim-eep.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/07a-sts-sim-eep.ipynb)
      * [07ba-sts-sim-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/07ba-sts-sim-eeap.ipynb)
      * [07bb-sts-sim-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/07bb-sts-sim-eeap.ipynb)
      * [07bc-sts-sim-eeap.ipynb](https://github.com/sujitpal/eeap-examples/blob/master/src/07bc-sts-sim-eeap.ipynb)


## Introduction

This repository contains some examples of applying the [Embed, Encode, Attend, Predict (EEAP)](https://explosion.ai/blog/deep-learning-formula-nlp) recipe proposed by Matthew Honnibal, creator of the [SpaCy](https://spacy.io/) deep learning toolkit, for building Deep Learning pipelines. 

I also gave a talk about this at [my talk at PyData Seattle 2017](https://pydata.org/seattle2017/schedule/presentation/84/).

Code is in Python. All models are built using the awesome [Keras](https://keras.io/) library. Supporting code uses [NLTK](http://www.nltk.org/) and [Scikit-Learn](http://scikit-learn.org/stable/).

The examples use 4 custom Attention layers, also available here as a Python include file. The examples themselves are written as [Jupyter notebooks](http://jupyter.org/).

A good complete implementation of attention can be found [here](https://github.com/wballard/mailscanner/blob/attention/mailscanner/layers/attention.py).


## Data

Please refer to [data/README.md](data/README.md) for instructions on how to download the data necessary to run these examples.


## Examples

### Document Classification Task

The document classification task attempts to build a classification model for documents by treating it as a sequence of sentences, and sentences as sequence of words. We start with the bag of words approach, computing document embeddings as an average of its sentence embeddings, and sentence embeddings as an average of its word embeddings. Next we build a hierarchical model for building sentence embeddings using a bidirectional LSTM, and embed this model within one that builds document embeddings by encoding the output of this model using another bidirectional LSTM. Finally we add attention layers to each level (sentence and document model). Our final model is depicted in the figure below:

<img src="docs/example-1-3.png"/>

The models were run against the Reuters 20 Newsgroups data in order to classify a given document into one of 20 classes. The chart below shows the results of running these different experiments. The interesting value here is the test set accuracy, but we have shown training and validation set accuracies as well for completeness.

<img src="docs/_results_1.png"/>

As you can see, the accuracy rises from about 71% for the bag of words model to about 82% for the hierarchical model that incorporates the Matrix Vector Attention models.

---

### Document Similarity Task

The Document Similarity task uses a nested model similar to the document classification task, where the sentence model generates a sentence embedding from a sequence of word embeddings, and a document model embeds the sentence model to generate a document embedding. A pair of such networks are set up to produce document vectors from the documents being compared, and the concatenated vector fed into a fully connected network to predict a binary (similar / not similar) outcome.

<img src="docs/example-2.png"/>

The dataset for this was manufactured from the Reuters 20 newsgroup dataset. TF-IDF vectors were generated for all 10,000 test set documents, and the similarity between all pairs of these vectors were calculated. Then the top 5 percentile was selected as the positive set and the bottom 5 percentile as the negative set. Even so, there does not appear to be too much differentiation, similarity values differed by about 0.2 between the two sets. A 1% sample was then drawn from either set to make the training set for this network.

We built two models, one without attention at either the sentence or document layer, and one with attention on the document layer. Results are shown below:

<img src="docs/_results_2.png"/>

---

### Sentence Similarity Task

The Sentence Similarity task uses the Semantic Similarity Task dataset from 2012. The objective is to classify a pair of sentences into a continuous scale of similarity from 0 to 5. We build a regression network as shown below. Our loss function is Mean Squared Error and Optimizer is RMSProp. Evaluation is done by computing the RMSE between the label similarity and the network predictions of the test set. In addition, we also compute the Pearson and Spearman (rank) correlations between the labels and predictions of the test set.

<img src="docs/example-3.png"/>

Our baseline is a hierarchical network that computes an encoding for each sentence in the pair, where the encodings without attention are used to generate the prediction. We compare the baseline to Matrix Matrix dot attention proposed by Parikh, et al where the inputs are scaled to \[-1, 1\] (MM-dot(s)). Next we compare with an unscaled version of this (MM-dot). Finally, we introduce two new attention implementations based on a description on [this Tensorflow NMT page](https://github.com/tensorflow/nmt) - specifically, an additive attention (MM-add) proposed by Bahdanau, et al, and a multiplicative attention (MM-mult) proposed by Luong, et al. Both operate on the encoder outputs without scaling via tanh. Results are shown below. As can be seen, the MM-add and MM-mult result in lower RMSE and generally higher Pearson and Spearman correlations than the baseline.

<img src="docs/_results_3.png"/>


