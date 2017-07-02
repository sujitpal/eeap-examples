# Data for eeap-examples

## Document Classification Task

The document classification task uses the [Reuters 20-newsgroup dataset](http://qwone.com/~jason/20Newsgroups/) provided via the scikit-learn datasets package, and classifies each document as 1 of 20 possible newsgroup labels.

We use GLoVe embeddings for the 840 billion words corpus to do the embedding part of our classification pipeline. To download this embedding, run the following commands (from the data directory (this one)):

    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip -a glove.840B.300d.zip
    rm glove.840B.300d.zip

## Document Similarity Task

The document similarity task tries to classify pairs of documents from the Reuters 20-newsgroup dataset as either similar or dissimilar. To build the dataset, we compute TF-IDF vectors from the test set in the Reuters 20-newsgroup data, and then compute similarity score on all pairs of these document vectors. We then consider the top 5 percentile as the positive set and the bottom 5 percentile as the negative set, and randomly sample 1% of these two sets to create the dataset.

To create the dataset, run the following code from the src directory.

    python ng-sim-datagen.py


## Sentence Similarity Task

The Sentence Similarity task uses the 2012 Semantic Similarity task dataset from [Semantic Textual Similarity Wiki](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page). The objective is to classify pairs of sentences into one of 5 different similarity ranges. To download this data, run the following commands (from the data (this) directory):

    wget http://ixa2.si.ehu.es/stswiki/images/e/e4/STS2012-en-train.zip
    wget http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip
    unzip -a STS2012-en-train.zip
    unzip -a STS2012-en-test.zip

