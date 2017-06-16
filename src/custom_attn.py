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
from keras.layers.wrappers import Bidirectional, TimeDistributed
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
        att = AttentionM()(enc)

    """    

    def __init__(self, **kwargs):
        super(AttentionM, self).__init__(**kwargs)

    
    def build(self, input_shape):
        # W: (BATCH_SIZE, EMBED_SIZE, 1)
        # b: (BATCH_SIZE, MAX_TIMESTEPS, 2)
        self.W = self.add_weight(name="W_{:s}".format(self.name), 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1],),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)


    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # alpha: (BATCH_SIZE, MAX_TIMESTEPS)
        ht = K.tanh(K.squeeze(K.dot(x, self.W), 2) + self.b)
        at = K.softmax(ht)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # alpha: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)        

    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
        

class AttentionMV(Layer):
    
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a context vector. Context vector is optional, if supplied it 
    will be used, otherwise a context vector will be generated.
    
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ctx - 2D Tensor of shape (BATCH_SIZE, EMBED_SIZE) (optional)
        
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)

        # with user supplied vector
        ctx = GlobalAveragePooling1D()(enc)
        att = AttentionMV()([enc, ctx])
        
        # without user supplied vector
        att = AttentionMV()([enc])
    
    """
    
    def __init__(self, **kwargs):
        super(AttentionMV, self).__init__(**kwargs)

        
    def build(self, input_shape):
        if type(input_shape) is list:
            self.generate_vector = False
            embed_size = input_shape[0][-1]
        else:
            self.generate_vector = True
            embed_size = input_shape[-1]
            
        # W: (BATCH_SIZE, EMBED_SIZE, EMBED_SIZE)
        # b: (BATCH_SIZE, EMBED_SIZE)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(embed_size, embed_size),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(embed_size,),
                                 initializer="zeros")

        if self.generate_vector:
            # u: (BATCH_SIZE, EMBED_SIZE, 1)
            self.u = self.add_weight(name="u_{:s}".format(self.name),
                                     shape=(input_shape[-1], 1),
                                     initializer="normal")
            
        super(AttentionMV, self).build(input_shape)


    def call(self, xs, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS(+1), EMBED_SIZE)
        if self.generate_vector:
            x = xs[0]
        else:
            x, self.u = xs
            if type(mask) is list:
                mask = mask[0]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # u: (BATCH_SIZE, EMBED_SIZE)
        # ht: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ht = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.dot(ht, self.u))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, eEMBED_SIZE)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        if self.generate_vector:
            return (input_shape[0], input_shape[-1])
        else:
            return (input_shape[0][0], input_shape[0][-1])



class AttentionMM(Layer):

    """
    Keras layer to compute an attention vector on a pair of incoming
    matrices.
    
    # Input
        m1 - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        m2 - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        merge_mode - one of concat, diff, prod, avg or max.
        
    # Output
        if merge_mode == "concat":
            2D Tensor of shape (BATCH_SIZE, EMBED_SIZE*2) 
        else:
            2D Tensor of shape (BATCH_SIZE, EMBED_SIZE) 

    # Usage
        enc1 = LSTM(EMBED_SIZE, return_sequences=True)(...)
        enc2 = LSTM(EMBED_SIZE, return_sequences=True)(...)
        
        att = AttentionMM("concat")([enc1, enc2])

    """
    
    def __init__(self, merge_mode, **kwargs):
        self.merge_mode = merge_mode
        assert self.merge_mode in set(["concat", "diff", "prod", "avg", "max"])
        super(AttentionMM, self).__init__(**kwargs)


    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        assert input_shape[0] == input_shape[1]
        # W1: (BATCH_SIZE, EMBED_SIZE, 1)
        # b1: (BATCH_SIZE, MAX_TIMESTEPS)
        # W2: (BATCH_SIZE, EMBED_SIZE, 1)
        # b2: (BATCH_SIZE, MAX_TIMESTEPS)
        self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                  shape=(input_shape[0][-1], 1),
                                  initializer="normal")
        self.b1 = self.add_weight(name="b1_{:s}".format(self.name),
                                  shape=(input_shape[0][1],),
                                  initializer="zeros")
        self.W2 = self.add_weight(name="W2_{:s}".format(self.name),
                                  shape=(input_shape[1][-1], 1),
                                  initializer="normal")
        self.b2 = self.add_weight(name="b2_{:s}".format(self.name),
                                  shape=(input_shape[1][1],),
                                  initializer="zeros")
        super(AttentionMM, self).build(input_shape)


    def call(self, xs, mask=None):
        assert len(xs) == 2
        # separate out input matrices
        # x1.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # x2.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        x1, x2 = xs
        # build alignment matrix 
        # align: (BATCH_SIZE, MAX_TIMESTEPS, MAX_TIMESTEPS)
        align = K.batch_dot(x1, x2, axes=(2, 2))
        # ht1: (BATCH_SIZE, MAX_TIMESTEPS)
        # ht2: (BATCH_SIZE, MAX_TIMESTEPS)
        ht1 = K.tanh(K.dot(x1, self.W1) + self.b1)
        ht2 = K.tanh(K.dot(x2, self.W2) + self.b2)
        # at1: (BATCH_SIZE, MAX_TIMESTEPS)
        # at2: (BATCH_SIZE, MAX_TIMESTEPS)
        at1 = K.softmax(K.batch_dot(ht2, align))
#        if mask[0] is not None:
#            at1 *= K.cast(mask[0], K.floatx())
        at1 = K.sum(at1, axis=1)
        at2 = K.softmax(K.batch_dot(ht1, 
                K.permute_dimensions(align, (0, 2, 1))))
#        if mask[1] is not None:
#            at2 *= K.cast(mask[1], K.floatx())
        at2 = K.sum(at2, axis=1)
        # ot1: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # ot2: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot1 = x1 * at1
        ot2 = x2 * at2
        # o1: (BATCH_SIZE, EMBED_SIZE)
        # o2: (BATCH_SIZE, EMBED_SIZE)
        o1 = K.sum(ot1, axis=1)
        o2 = K.sum(ot2, axis=1)
        # merge the attention vectors according to merge_mode
        if self.merge_mode == "concat":
            return concatenate([o1, o2], axis=1)
        elif self.merge_mode == "diff":
            return add([o1, -o2])
        elif self.merge_mode == "prod":
            return multiply([o1, o2])
        elif self.merge_mode == "avg":
            return average([o1, o2])
        else: # max
            return maximum([o1, o2])


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    
    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        if self.merge_mode == "concat":
            # output shape: (BATCH_SIZE, EMBED_SIZE*2)
            return (input_shape[0][0], input_shape[0][2] * 2)
        else:
            # output shape: (BATCH_SIZE, EMBED_SIZE)
            return (input_shape[0][0], input_shape[0][2])
            

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

    sent_att = AttentionM()(sent_enc)

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
                                 
    sent_att = AttentionM()(sent_enc)

    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    sent_model = Model(inputs=sent_inputs, outputs=sent_pred)
    
    # document pipeline    
    doc_inputs = Input(shape=(max_sents, max_words), dtype="int32")

    doc_emb = TimeDistributed(sent_model)(doc_inputs)

    doc_enc = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb)
    
    doc_att = AttentionM()(doc_enc)
    
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
    sent_att = AttentionMV()([sent_enc])
    
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
    
    sent_att = AttentionMV()([sent_enc, sent_vec])

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

    sent_att = AttentionMV()([sent_enc])

    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    sent_model = Model(inputs=sent_inputs, outputs=sent_pred)
    
    # document pipeline    
    doc_inputs = Input(shape=(max_sents, max_words), dtype="int32")

    doc_emb = TimeDistributed(sent_model)(doc_inputs)

    doc_enc = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb)

    doc_att = AttentionMV()([doc_enc])
    
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
    sent_att = AttentionMV()([sent_enc, sent_vec])

    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)
    
    sent_model = Model(inputs=sent_inputs, outputs=sent_pred)
    
    # document pipeline
    doc_inputs = Input(shape=(max_sents, max_words), dtype="int32")
    
    doc_emb = TimeDistributed(sent_model)(doc_inputs)
    
    doc_enc = Bidirectional(GRU(doc_embed_size, 
                                return_sequences=True))(doc_emb)
    
    doc_vec = GlobalAveragePooling1D()(doc_enc)
    doc_att = AttentionMV()([doc_enc, doc_vec])
    
    fc1_dropout = Dropout(0.2)(doc_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    doc_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=doc_inputs, outputs=doc_pred)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model.summary()



def test_attention_mm1(batch_size, word_embed_size, sent_embed_size,
                      num_classes, max_words, vocab_size):
    """ model summary (matrix-matrix attention), similarity network """

    E = np.random.random((vocab_size, word_embed_size))

    # LHS sentence    
    sent_inputs_left = Input(shape=(max_words,), dtype="int32")
    sent_emb_left = Embedding(input_dim=vocab_size,
                              output_dim=word_embed_size,
                              mask_zero=True,
                              weights=[E])(sent_inputs_left)
    sent_enc_left = Bidirectional(GRU(sent_embed_size,
                                      return_sequences=True))(sent_emb_left)
    
    # RHS sentence
    sent_inputs_right = Input(shape=(max_words,), dtype="int32")
    sent_emb_right = Embedding(input_dim=vocab_size,
                               output_dim=word_embed_size,
                               mask_zero=True,
                               weights=[E])(sent_inputs_right)
    sent_enc_right = Bidirectional(GRU(sent_embed_size,
                                       return_sequences=True))(sent_emb_right)

    # attention
    sent_att = AttentionMM("concat")([sent_enc_left, sent_enc_right])

    # prediction    
    fc1_dropout = Dropout(0.2)(sent_att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.2)(fc1)
    sent_pred = Dense(num_classes, activation="softmax")(fc2_dropout)

    model = Model(inputs=[sent_inputs_left, sent_inputs_right], 
                         outputs=sent_pred)
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

#    print(test_attention_m1.__doc__)
#    print(test_attention_m1(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
#                           MAX_WORDS, VOCAB_SIZE))
#    
#    print(test_attention_m2.__doc__)
#    print(test_attention_m2(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
#                     DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
#                     NUM_CLASSES))
#
#    print(test_attention_mv1.__doc__)
#    print(test_attention_mv1(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
#                            MAX_WORDS, VOCAB_SIZE))
#
#    print(test_attention_mv2.__doc__)
#    print(test_attention_mv2(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
#                            MAX_WORDS, VOCAB_SIZE))
#
#    print(test_attention_mv3.__doc__)
#    print(test_attention_mv3(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
#                             DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
#                             NUM_CLASSES))
#
#    print(test_attention_mv4.__doc__)
#    print(test_attention_mv4(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE,
#                             DOC_EMBED_SIZE, VOCAB_SIZE, MAX_WORDS, MAX_SENTS,
#                             NUM_CLASSES))

    print(test_attention_mm1.__doc__)
    print(test_attention_mm1(BATCH_SIZE, WORD_EMBED_SIZE, SENT_EMBED_SIZE, NUM_CLASSES,
                            MAX_WORDS, VOCAB_SIZE))


if __name__ == "__main__":
    run_tests()

