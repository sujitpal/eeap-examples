from __future__ import division, print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, average, concatenate, maximum, multiply
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
import numpy as np


class AttentionMV(Layer):
    
    def __init__(self, m_shape, v_shape, **kwargs):
        self.m_shape = m_shape
        self.v_shape = v_shape
        super(AttentionMV, self).__init__(**kwargs)

        
    def build(self, input_shape):
        super(AttentionMV, self).build(input_shape)


    def call(self, x, mask=None):
        assert len(x) == 2
        # split up into matrix and vector
        # M.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # V.shape == (BATCH_SIZE, EMBED_SIZE)
        M = x[0]
        # reshape V to (BATCH_SIZE, EMBED_SIZE, MAX_TIMESTEPS) by 
        # adding single timestep dimension and repeating
        V = K.repeat_elements(K.expand_dims(x[1], axis=2), 
                              self.m_shape[1], axis=2)
        # compute alpha weight variable (BATCH_SIZE, MAX_TIMESTEPS, MAX_TIMESTEPS)
        alpha = K.softmax(K.batch_dot(M, V))
        # multiply inputs with the given weights and produce attended
        attended = K.sum(K.batch_dot(K.permute_dimensions(M, (0, 2, 1)), 
                                     alpha), axis=2)
        return attended


    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (self.v_shape[0], self.v_shape[1])


class AttentionMM(Layer):
    
    def __init__(self, m1_shape, m2_shape, merge_mode, **kwargs):
        self.m1_shape = m1_shape
        self.m2_shape = m2_shape
        self.merge_mode = merge_mode
        assert self.merge_mode in set(["concat", "diff", "prod", "avg", "max"])
        super(AttentionMM, self).__init__(**kwargs)


    def build(self, input_shape):
        super(AttentionMM, self).build(input_shape)


    def call(self, x, mask=None):
        assert len(x) == 2
        # separate out input matrices
        # M1.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # M2.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        M1, M2 = x[0], x[1]
        print("M1=", M1)
        print("M2=", M2)
        # compute alpha weight variable (BATCH_SIZE, MAX_TIMESTEPS, MAX_TIMESTEPS)
        alpha = K.softmax(K.batch_dot(M1, K.permute_dimensions(M2, (0, 2, 1))))
        # multiply inputs with the given weights and produce attended vectors
        # a1.shape == (BATCH_SIZE, EMBED_SIZE)
        # a2.shape == (BATCH_SIZE, EMBED_SIZE)
        a1 = K.sum(K.batch_dot(M1, alpha), axis=1)
        a2 = K.sum(K.batch_dot(M1, K.permute_dimensions(alpha, (0, 2, 1))), 
                   axis=1)
        # merge the attention vectors according to merge_mode
        if self.merge_mode == "concat":
            return concatenate([a1, a2], axis=1)
        elif self.merge_mode == "diff":
            return add([a1, -a2])
        elif self.merge_mode == "prod":
            return multiply([a1, a2])
        elif self.merge_mode == "avg":
            return average([a1, a2])
        else: # max
            return maximum([a1, a2])

    
    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        if self.merge_mode == "concat":
            # output shape: (BATCH_SIZE, EMBED_SIZE*2)
            return (self.m1_shape[0], self.m1_shape[2] * 2)
        else:
            # output shape: (BATCH_SIZE, EMBED_SIZE)
            return (self.m1_shape[0], self.m1_shape[2])
            

def test_attention_mv(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):

    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size, 
                                 return_sequences=True))(sent_emb)

    # generate summary vector
    sent_sum = GlobalMaxPooling1D()(sent_enc)
    
    m_shape = (batch_size, max_words, sent_embed_size * 2)
    v_shape = (batch_size, sent_embed_size * 2)
    sent_attn = AttentionMV(m_shape, v_shape)([sent_enc, sent_sum])

    fc1_dropout = Dropout(0.2)(sent_attn)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=sent_inputs, outputs=sent_pred)
    return model.summary()


def test_attention_mm(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):
    
    E = np.random.random((vocab_size, word_embed_size))

    # LHS sentence    
    sent_inputs_left = Input(shape=(max_words,), dtype="int32")
    sent_emb_left = Embedding(input_dim=vocab_size,
                              output_dim=word_embed_size,
                              weights=[E])(sent_inputs_left)
    sent_enc_left = Bidirectional(GRU(sent_embed_size,
                                      return_sequences=True))(sent_emb_left)
    
    # RHS sentence
    sent_inputs_right = Input(shape=(max_words,), dtype="int32")
    sent_emb_right = Embedding(input_dim=vocab_size,
                               output_dim=word_embed_size,
                               weights=[E])(sent_inputs_right)
    sent_enc_right = Bidirectional(GRU(sent_embed_size,
                                       return_sequences=True))(sent_emb_right)

    # attention
    m1_shape = (batch_size, max_words, sent_embed_size * 2)
    m2_shape = (batch_size, max_words, sent_embed_size * 2)
    sent_attn = AttentionMM(m1_shape, m2_shape, 
                            "concat")([sent_enc_left, sent_enc_right])

    # prediction    
    fc1_dropout = Dropout(0.2)(sent_attn)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=[sent_inputs_left, sent_inputs_right], 
                         outputs=sent_pred)
    return model.summary()

    
def run_tests():
    BATCH_SIZE = 32
    WORD_EMBED_SIZE = 300
    SENT_EMBED_SIZE = 200
    NUM_CLASSES = 20
    MAX_WORDS = 60
    VOCAB_SIZE = 50000

    print("model summary (matrix-vector attention)")
    print(test_attention_mv(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))
    print("model summary (matrix-matrix attention)")
    print(test_attention_mm(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))


if __name__ == "__main__":
    run_tests()

