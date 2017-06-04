# -*- coding: utf-8 -*-
# Adapted from: Out of core classification of Text Documents
# from the scikit-learn documentation.
# http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
# Converts the reuters-21578 SGML files to the following set of flat files:
#   * rt-vocab.tsv -- term and frequency in corpus, tab separated
#   * rt-text.tsv -- document-id, text, tab separated.
#   * rt-sent.tsv -- document-id, sentence-id, sentence, tab separated
#   * rt-tags.tsv -- document-id, comma-separated list of tags, tab separated
#
from __future__ import division, print_function
from sklearn.externals.six.moves import html_parser
from glob import glob
import collections
import nltk
import os
import re

class ReutersParser(html_parser.HTMLParser):
    """ Utility class to parse a SGML file and yield documents one at 
        a time. 
    """
    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(reuters_dir):
    """ Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """
    parser = ReutersParser()
    for filename in glob(os.path.join(reuters_dir, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc


##################### main ######################

DATA_DIR = "../data"
REUTERS_DIR = os.path.join(DATA_DIR, "reuters-21578")
VOCAB_SIZE = None

ftext = open(os.path.join(DATA_DIR, "rt-text.tsv"), "wb")
fsent = open(os.path.join(DATA_DIR, "rt-sent.tsv"), "wb")
ftags = open(os.path.join(DATA_DIR, "rt-tags.tsv"), "wb")
num_read = 0
counter = collections.Counter()
for doc in stream_reuters_documents(REUTERS_DIR):
    # periodic heartbeat report
    if num_read % 100 == 0:
        print("building features from {:d} docs".format(num_read))
    # skip docs without specified topic
    topics = doc["topics"]
    if len(topics) == 0:
        continue
    title = doc["title"]
    body = doc["body"]
    num_read += 1
    # concatenate title and body and convert to list of word indexes
    title_body = ". ".join([title, body]).lower()
    title_body = re.sub("\n", "", title_body)
    title_body = title_body.encode("utf8").decode("ascii", "ignore")
    num_sent = 0
    for sent in nltk.sent_tokenize(title_body):
        for word in nltk.word_tokenize(sent):
            counter[word] += 1
        fsent.write("{:d}\t{:d}\t{:s}\n".format(num_read, num_sent, sent))
        num_sent += 1
    ftext.write("{:d}\t{:s}\n".format(num_read, title_body))
    ftags.write("{:d}\t{:s}\n".format(num_read, ",".join(topics)))

ftext.close()
fsent.close()
ftags.close()

fvocab = open(os.path.join(DATA_DIR, "rt-vocab.tsv"), "wb")
for word, count in counter.most_common(VOCAB_SIZE):
    fvocab.write("{:s}\t{:d}\n".format(word, count))
fvocab.close()

print("features built from {:d} docs, complete".format(num_read))
