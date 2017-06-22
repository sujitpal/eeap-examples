# eeap-20newsgroup-classify-example

An example of applying the [Embed, Encode, Attend, Predict (EEAP)](https://explosion.ai/blog/deep-learning-formula-nlp) formula for Deep Learning NLP models proposed by Matthew Honnibal, creator of the [SpaCy](https://spacy.io/) deep learning toolkit. Our example is to classify newsgroup message category from the [20 newsgroups dataset](http://qwone.com/~jason/20Newsgroups/).

## Data

### Classification Task

The classification task uses the Reuters 20-newsgroups data, as provided via the scikit-learn datasets package. Here we download the GloVe embeddings for the 840 billion word corpus.

    mkdir data
    cd data
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip -a glove.840B.300d.zip
    rm glove.840B.300d.zip

### Document Similarity Task

From [Semantic Textual Similarity Wiki](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page).

    cd data
    wget http://ixa2.si.ehu.es/stswiki/images/e/e4/STS2012-en-train.zip
    wget http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip
    unzip -a STS2012-en-train.zip
    unzip -a STS2012-en-test.zip

