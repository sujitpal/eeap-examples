# e3p-docsim-example

An example of applying the [Embed, Encode, Attend, Predict (EEAP)](https://explosion.ai/blog/deep-learning-formula-nlp) formula for Deep Learning NLP models proposed by Matthew Honnibal, creator of the [SpaCy](https://spacy.io/) deep learning toolkit. Our example is to predict document similarity between documents in the [Reuters-21578 corpus](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection).

## Running the code

* Make a data folder under the project directory.
* Download and expand the Reuters-21578 corpus into this folder. This will create a data/reuters-21578 folder under the project directory.
* Run the parse-reuters.py script, this will parse the corpus data and produce two flat files, one for text and another for tags in the data directory, called text.tsv and tags.tsv respectively.
 
