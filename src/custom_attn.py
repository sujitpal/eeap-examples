# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add, average, concatenate, maximum, multiply


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
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name), 
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)


    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
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


class AttentionMC(Layer):
    
    """
    Keras layer to compute an attention vector on an incoming matrix
    using a learned context vector.
    
    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        
    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)
        att = AttentionMC()(enc)
    
    """
    
    def __init__(self, **kwargs):
        super(AttentionMC, self).__init__(**kwargs)

        
    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # u: (MAX_TIMESTEPS, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer="normal")
        super(AttentionMC, self).build(input_shape)


    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.dot(et, self.u))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0], input_shape[-1])


class AttentionMV(Layer):
    
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a user provided context vector.
    
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
    
    """
    
    def __init__(self, **kwargs):
        super(AttentionMV, self).__init__(**kwargs)

        
    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # U: (EMBED_SIZE, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[0][-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[0][1], 1),
                                 initializer="zeros")
        self.U = self.add_weight(name="U_{:s}".format(self.name),
                                 shape=(input_shape[0][-1],
                                        input_shape[0][1]),
                                 initializer="normal")
        super(AttentionMV, self).build(input_shape)


    def call(self, xs, mask=None):
        # input: [x, u]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # u: (BATCH_SIZE, EMBED_SIZE)
        x, c = xs
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.dot(c, self.U) + K.squeeze((K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)


    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None


    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
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
        # W1: (EMBED_SIZE, 1)
        # b1: (MAX_TIMESTEPS, 1)
        # W2: (EMBED_SIZE, 1)
        # b2: (MAX_TIMESTEPS, 1)
        self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                  shape=(input_shape[0][-1], 1),
                                  initializer="normal")
        self.b1 = self.add_weight(name="b1_{:s}".format(self.name),
                                  shape=(input_shape[0][1], 1),
                                  initializer="zeros")
        self.U1 = self.add_weight(name="U1_{:s}".format(self.name), 
                                  shape=(input_shape[0][-1], input_shape[0][1]),
                                  initializer="normal")
        self.W2 = self.add_weight(name="W2_{:s}".format(self.name),
                                  shape=(input_shape[1][-1], 1),
                                  initializer="normal")
        self.b2 = self.add_weight(name="b2_{:s}".format(self.name),
                                  shape=(input_shape[1][1], 1),
                                  initializer="zeros")
        self.U2 = self.add_weight(name="U2_{:s}".format(self.name), 
                                  shape=(input_shape[1][-1], input_shape[1][1]),
                                  initializer="normal")
        super(AttentionMM, self).build(input_shape)


    def call(self, xs, mask=None):
        assert len(xs) == 2
        # separate out input matrices
        # x1.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # x2.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        x1, x2 = xs
        # build alignment matrix 
        # S: (BATCH_SIZE, MAX_TIMESTEPS, MAX_TIMESTEPS)
        S = K.batch_dot(x1, x2, axes=(2, 2))
        # build context vectors
        # c1: (BATCH_SIZE, EMBED_SIZE)
        # c2: (BATCH_SIZE, EMBED_SIZE)
        c1 = K.mean(K.batch_dot(S, x2, axes=(1, 1)), axis=1)
        c2 = K.mean(K.batch_dot(K.permute_dimensions(S, (0, 2, 1)), 
                                x1, axes=(1, 1)), axis=1)
        # combine context with inputs
        # et1: (BATCH_SIZE, MAX_TIMESTEPS)
        # et2: (BATCH_SIZE, MAX_TIMESTEPS)
        et1 = K.dot(c1, self.U1) + K.squeeze((K.dot(x1, self.W1) + self.b1), 
                    axis=-1)
        et2 = K.dot(c2, self.U2) + K.squeeze((K.dot(x2, self.W2) + self.b2), 
                    axis=-1)
        # at1: (BATCH_SIZE, MAX_TIMESTEPS)
        # at2: (BATCH_SIZE, MAX_TIMESTEPS)
        at1 = K.softmax(et1)
        at2 = K.softmax(et2)
        if mask is not None and mask[0] is not None:
            at1 *= K.cast(mask[0], K.floatx())
        if mask is not None and mask[1] is not None:
            at2 *= K.cast(mask[1], K.floatx())
        # o1: (BATCH_SIZE, EMBED_SIZE)
        # o2: (BATCH_SIZE, EMBED_SIZE)
        atx1 = K.expand_dims(at1, axis=-1)
        atx2 = K.expand_dims(at2, axis=-1)
        o1 = K.sum(x1 * atx1, axis=1)
        o2 = K.sum(x2 * atx2, axis=1)
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
        if self.merge_mode == "concat":
            # output shape: (BATCH_SIZE, EMBED_SIZE*2)
            return (input_shape[0][0], input_shape[0][2] * 2)
        else:
            # output shape: (BATCH_SIZE, EMBED_SIZE)
            return (input_shape[0][0], input_shape[0][2])
            

