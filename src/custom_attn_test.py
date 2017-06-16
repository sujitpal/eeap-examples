# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
import numpy as np

import custom_attn


###############################################################################

def test_attention_m1(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):
    """ model summary (matrix attention) -- sentence only """

    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         mask_zero=True,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)

    sent_att = custom_attn.AttentionM()(sent_enc)

    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=sent_inputs, outputs=sent_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()


def test_attention_m2(batch_size, word_embed_size, sent_embed_size,
                     doc_embed_size, vocab_size, max_words, max_sents,
                     num_classes):
    """ model summary (matrix attention) -- full """

    # sentence encoder
    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         mask_zero=True,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)
                                 
    sent_att = custom_attn.AttentionM()(sent_enc)

    sent_model = Model(inputs=sent_inputs, outputs=sent_att)
    
    # document pipeline    
    doc_inputs = Input(shape=(max_sents, max_words), dtype="int32")

    doc_emb = TimeDistributed(sent_model)(doc_inputs)

    doc_enc = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb)
    
    doc_att = custom_attn.AttentionM()(doc_enc)
    
    fc1_dropout = Dropout(0.2)(doc_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    doc_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=doc_inputs, outputs=doc_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()


def test_attention_mv1(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):
    """ model summary (matrix-vector attention) -- sentence only, w/o query """

    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)

    # generate summary vector
    sent_att = custom_attn.AttentionMV()([sent_enc])
    
    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)
    
    model = Model(inputs=sent_inputs, outputs=sent_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()


def test_attention_mv2(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):

    """ model summary (matrix-vector attention) -- sentence only, w/ query """

    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)

    # generate summary vector
    sent_vec = GlobalAveragePooling1D()(sent_enc)
    
    sent_att = custom_attn.AttentionMV()([sent_enc, sent_vec])

    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=sent_inputs, outputs=sent_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()


def test_attention_mv3(batch_size, word_embed_size, sent_embed_size,
                       doc_embed_size, vocab_size, max_words, max_sents,
                       num_classes):
    """ model summary (matrix-vector attention) -- full, w/o query """

    # sentence encoder
    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         mask_zero=True,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)

    sent_att = custom_attn.AttentionMV()([sent_enc])

    sent_model = Model(inputs=sent_inputs, outputs=sent_att)
    
    # document pipeline    
    doc_inputs = Input(shape=(max_sents, max_words), dtype="int32")

    doc_emb = TimeDistributed(sent_model)(doc_inputs)

    doc_enc = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb)

    doc_att = custom_attn.AttentionMV()([doc_enc])
    
    fc1_dropout = Dropout(0.2)(doc_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    doc_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=doc_inputs, outputs=doc_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()


def test_attention_mv4(batch_size, word_embed_size, sent_embed_size,
                       doc_embed_size, vocab_size, max_words, max_sents,
                       num_classes):
    """ model summary (matrix-vector attention) -- full, w/ query """

    # sentence encoder
    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)

    sent_vec = GlobalAveragePooling1D()(sent_enc)
    sent_att = custom_attn.AttentionMV()([sent_enc, sent_vec])

    sent_model = Model(inputs=sent_inputs, outputs=sent_att)
    
    # document pipeline
    doc_inputs = Input(shape=(max_sents, max_words), dtype="int32")
    
    doc_emb = TimeDistributed(sent_model)(doc_inputs)
    
    doc_enc = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb)
    
    doc_vec = GlobalAveragePooling1D()(doc_enc)
    doc_att = custom_attn.AttentionMV()([doc_enc, doc_vec])
    
    fc1_dropout = Dropout(0.2)(doc_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    doc_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=doc_inputs, outputs=doc_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()

def test_attention_mm1(batch_size, word_embed_size, sent_embed_size,
                       doc_embed_size, vocab_size, max_words, max_sents,
                       num_classes):
    """ model summary (matrix-matrix attention), similarity network """

    def sum_over_axis(X, axis):
        return K.mean(X, axis=axis)

    E = np.random.random((vocab_size, word_embed_size))

    # LHS sentence    
    sent_in_left = Input(shape=(max_words,), dtype="int32")
    sent_emb_left = Embedding(input_dim=vocab_size,
                              output_dim=word_embed_size,
                              weights=[E])(sent_in_left)
    sent_enc_left = Bidirectional(GRU(sent_embed_size,
                                      return_sequences=True))(sent_emb_left)
                                      
    sent_model_left = Model(inputs=sent_in_left, outputs=sent_enc_left)                                      

    # RHS sentence
    sent_in_right = Input(shape=(max_words,), dtype="int32")
    sent_emb_right = Embedding(input_dim=vocab_size,
                               output_dim=word_embed_size,
                               weights=[E])(sent_in_right)
    sent_enc_right = Bidirectional(GRU(sent_embed_size,
                                       return_sequences=True))(sent_emb_right)

    sent_model_right = Model(inputs=sent_in_right, outputs=sent_enc_right)
                                      
    # LHS document
    doc_in_left = Input(shape=(max_sents, max_words), dtype="int32")
    
    doc_emb_left = TimeDistributed(sent_model_left)(doc_in_left)
    doc_emb_left = Lambda(lambda x: sum_over_axis(x, 2))(doc_emb_left)

    doc_enc_left = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb_left)
    
    # RHS document
    doc_in_right = Input(shape=(max_sents, max_words), dtype="int32")
    
    doc_emb_right = TimeDistributed(sent_model_right)(doc_in_right)
    doc_emb_right = Lambda(lambda x: sum_over_axis(x, 2))(doc_emb_right)

    doc_enc_right = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb_right)

    # attention
    doc_att = custom_attn.AttentionMM("concat")([doc_enc_left, doc_enc_right])

    # prediction    
    fc1_dropout = Dropout(0.2)(doc_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    doc_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=[doc_in_left, doc_in_right], outputs=doc_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()



    
def run_tests():

    BATCH_SIZE = 32

    WORD_EMBED_SIZE = 300
    SENT_EMBED_SIZE = 200
    DOC_EMBED_SIZE = 50

    VOCAB_SIZE = 50000

    NUM_CLASSES = 20
    MAX_WORDS = 60
    MAX_SENTS = 40

    print(test_attention_m1.__doc__)
    print(test_attention_m1(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                           MAX_WORDS, VOCAB_SIZE))
    
    print(test_attention_m2.__doc__)
    print(test_attention_m2(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
                     DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
                     NUM_CLASSES))

    print(test_attention_mv1.__doc__)
    print(test_attention_mv1(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))

    print(test_attention_mv2.__doc__)
    print(test_attention_mv2(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))

    print(test_attention_mv3.__doc__)
    print(test_attention_mv3(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
                             DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
                             NUM_CLASSES))

    print(test_attention_mv4.__doc__)
    print(test_attention_mv4(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
                             DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
                             NUM_CLASSES))

    print(test_attention_mm1.__doc__)
    print(test_attention_mm1(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
                             DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
                             NUM_CLASSES))


if __name__ == "__main__":
    run_tests()


