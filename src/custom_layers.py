from __future__ import division, print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, average, concatenate, maximum, multiply
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
import numpy as np



class AttentionM(Layer):
    
    """
    Keras layer to compute an attention vector on an incoming matrix.
    
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM((BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE))(enc)

    """    

    def __init__(self, m_shape, **kwargs):
        self.m_shape = m_shape
        super(AttentionM, self).__init__(**kwargs)

    
    def build(self, input_shape):
        assert (self.m_shape[1] == input_shape[1] and
                self.m_shape[2] == input_shape[2])
        # W: (BATCH_SIZE, EMBED_SIZE, 1)
        # b: (BATCH_SIZE, MAX_TIMESTEPS)
        self.W = K.random_normal_variable(
                shape=(self.m_shape[0], self.m_shape[-1], 1), 
                mean=0.0, scale=0.05)
        self.b = K.zeros((self.m_shape[0], self.m_shape[1]))
        super(AttentionM, self).build(input_shape)


    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # alpha: (BATCH_SIZE, MAX_TIMESTEPS)
        alpha = K.softmax(K.tanh(K.batch_dot(x, self.W) + self.b))
        if mask is not None:
            alpha *= K.cast(mask, K.floatx())
        # output: (BATCH_SIZE, EMBED_SIZE)
        alpha_emb = K.expand_dims(alpha, axis=-1)
        return K.sum(x * alpha_emb, axis=1)

    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
        
        

class AttentionMV(Layer):
    
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a context vector.
    
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ctx - 2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)
        
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)
        ctx = GlobalAveragePooling1D()(enc)
        m_shape = (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        v_shape = (BATCH_SIZE, EMBED_SIZE)
        att = AttentionMV(m_shape, v_shape)([enc, ctx])
    
    """
    
    def __init__(self, m_shape, v_shape, **kwargs):
        self.m_shape = m_shape
        self.v_shape = v_shape
        self.W = K.random_normal_variable
        super(AttentionMV, self).__init__(**kwargs)

        
    def build(self, input_shape):
        # W: (BATCH_SIZE, EMBED_SIZE, EMBED_SIZE)
        # b: (BATCH_SIZE, EMBED_SIZE)
        self.W = K.random_normal_variable(
                shape=(self.m_shape[0], self.m_shape[-1], self.m_shape[-1]), 
                mean=0.0, scale=0.05)
        self.b = K.zeros((self.m_shape[0], self.m_shape[-1]))
        super(AttentionMV, self).build(input_shape)


    def call(self, x, mask=None):
        assert len(x) == 2
        # split up into matrix and vector
        # m.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # v.shape == (BATCH_SIZE, EMBED_SIZE)
        m, v = x[0], x[1]
        # combine with weights to get u
        # u.shape == (BATCH_SIZE, EMBED_SIZE)
        u = K.tanh(K.batch_dot(m, self.W) + self.b)
        # compute softmax weights alpha
        #   as dot product of v.T and u (EMBED_SIZE, BATCH_SIZE, EMBED_SIZE)
        #   transpose indexes to (BATCH_SIZE, EMBED_SIZE, EMBED_SIZE)
        #   compute softmax
        alpha = K.softmax(K.permute_dimensions(
                K.dot(K.transpose(v), u), (1, 0, 2)))
        if mask is not None:
            alpha *= K.cast(mask, K.floatx())
        # compute output = weighted input across all timesteps
        # output.shape == (BATCH_SIZE, EMBED_SIZE)
        return K.sum(K.batch_dot(m, alpha), axis=1)


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (self.v_shape[0], self.v_shape[1])



class AttentionMM(Layer):

    """
    Keras layer to compute an attention vector on a pair of incoming
    matrices.
    
    # Input
        m1_shape - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        m2_shape - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        merge_mode - one of concat, diff, prod, avg or max.
        
    # Output
        if merge_mode == "concat":
            2D Tensor of shape (BATCH_SIZE, EMBED_SIZE*2) 
        else:
            2D Tensor of shape (BATCH_SIZE, EMBED_SIZE) 

    # Usage
        enc1 = LSTM(EMBED_SIZE, return_sequences=True)(...)
        enc2 = LSTM(EMBED_SIZE, return_sequences=True)(...)
        
        m_shape = (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        att = AttentionMM(m_shape, "concat")([enc1, enc2])

    """
    
    def __init__(self, m_shape, merge_mode, **kwargs):
        self.m_shape = m_shape
        self.merge_mode = merge_mode
        assert self.merge_mode in set(["concat", "diff", "prod", "avg", "max"])
        super(AttentionMM, self).__init__(**kwargs)


    def build(self, input_shape):
        # W1: (BATCH_SIZE, EMBED_SIZE, 1)
        self.W1 = K.random_normal_variable(
                shape=(self.m_shape[0], self.m_shape[-1], 1), 
                mean=0.0, scale=0.05)
        # b1: (BATCH_SIZE, MAX_TIMESTEPS)
        self.b1 = K.zeros((self.m_shape[0], self.m_shape[1]))
        # W2: (BATCH_SIZE, EMBED_SIZE, 1)
        self.W2 = K.random_normal_variable(
                shape=(self.m_shape[0], self.m_shape[-1], 1), 
                mean=0.0, scale=0.05)
        # b2: (BATCH_SIZE, MAX_TIMESTEPS)
        self.b2 = K.zeros((self.m_shape[0], self.m_shape[1]))
        super(AttentionMM, self).build(input_shape)


    def call(self, x, mask=None):
        assert len(x) == 2
        # separate out input matrices
        # m1.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # m2.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        m1, m2 = x[0], x[1]
        # build alignment matrix (BATCH_SIZE, MAX_TIMESTEPS, MAX_TIMESTEPS)
        align = K.batch_dot(m1, K.permute_dimensions(m2, (0, 2, 1))) 
        # u1.shape == (BATCH_SIZE, MAX_TIMESTEPS)
        # u2.shape == (BATCH_SIZE, MAX_TIMESTEPS)
        u1 = K.tanh(K.batch_dot(m1, self.W1) + self.b1)
        u2 = K.tanh(K.batch_dot(m2, self.W2) + self.b2)
        # alpha1.shape == (BATCH_SIZE, MAX_TIMESTEPS)
        # alpha2.shape == (BATCH_SIZE, MAX_TIMESTEPS)
        alpha1 = K.softmax(K.batch_dot(align, u2))
        alpha2 = K.softmax(K.batch_dot(
                K.permute_dimensions(align, (0, 2, 1)), u1))
        # v1.shape == (BATCH_SIZE, EMBED_SIZE)
        # v2.shape == (BATCH_SIZE, EMBED_SIZE)
        v1 = K.sum(K.batch_dot(K.permute_dimensions(m1, (0, 2, 1)), alpha1), 
                   axis=2)
        v2 = K.sum(K.batch_dot(K.permute_dimensions(m2, (0, 2, 1)), alpha2),
                   axis=2)
        # merge the attention vectors according to merge_mode
        if self.merge_mode == "concat":
            return concatenate([v1, v2], axis=1)
        elif self.merge_mode == "diff":
            return add([v1, -v2])
        elif self.merge_mode == "prod":
            return multiply([v1, v2])
        elif self.merge_mode == "avg":
            return average([v1, v2])
        else: # max
            return maximum([v1, v2])


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    
    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        if self.merge_mode == "concat":
            # output shape: (BATCH_SIZE, EMBED_SIZE*2)
            return (self.m_shape[0], self.m_shape[2] * 2)
        else:
            # output shape: (BATCH_SIZE, EMBED_SIZE)
            return (self.m_shape[0], self.m_shape[2])
            


def test_attention_m(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):

    E = np.random.random((vocab_size, word_embed_size))

    sent_inputs = Input(shape=(max_words,), dtype="int32")
    sent_emb = Embedding(input_dim=vocab_size,
                         output_dim=word_embed_size,
                         mask_zero=True,
                         weights=[E])(sent_inputs)
    sent_enc = Bidirectional(GRU(sent_embed_size,
                                 return_sequences=True))(sent_emb)

    m_shape = (batch_size, max_words, sent_embed_size * 2)
    sent_att = AttentionM(m_shape)(sent_enc)

    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=sent_inputs, outputs=sent_pred)
    return model.summary()
    

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
    sent_sum = GlobalAveragePooling1D()(sent_enc)
    
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
    m_shape = (batch_size, max_words, sent_embed_size * 2)
    sent_attn = AttentionMM(m_shape, "concat")([sent_enc_left, sent_enc_right])

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

    print("model summary (matrix attention)")
    print(test_attention_m(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                           MAX_WORDS, VOCAB_SIZE))

    print("model summary (matrix-vector attention)")
    print(test_attention_mv(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))

    print("model summary (matrix-matrix attention)")
    print(test_attention_mm(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))

if __name__ == "__main__":
    run_tests()

