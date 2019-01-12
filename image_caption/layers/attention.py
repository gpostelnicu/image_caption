from keras import backend as K
from keras import initializers
from keras.engine import Layer


class AttentionWeightedAverage(Layer):
    """
    Weighted average for image captioning. Inputs:
    - hidden state of shape (seqdim, seqlen)
    - visual features of shape (vfeat_dim, vfeatlen)

    The equation followed is (6) in https://arxiv.org/pdf/1612.01887.pdf

    z_t = w_h tanh(W_v V + (W_g h_t) 1_{vfeatlen}^T)
    \alpha_t = softmax(z_t)

    The dimension of the embedding, embed_dim means that:
    - w_h is of shape (vfeatlen, embed_dim)
    - W_v is of shape (embed_dim, vfeat_dim)
    - W_g is of shape (embed_dim, seqdim)
    """
    def __init__(self, embed_dim, **kwargs):
        self.kernel_initializer = initializers.get('uniform')
        self.supports_masking = True
        self.embed_dim = embed_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('An AttentionWeightedAverage layer should have 2 inputs.')
        if len(input_shape) < 2:
            raise ValueError('An AttentionWeightedAverage layer should have exactly 2 inputs.')

        assert len(input_shape[0]) == 2
        seqdim = input_shape[0][0]
        assert len(input_shape[1]) == 2
        self.vfeat_dim = input_shape[1][0]
        self.vfeat_len = input_shape[1][1]

        self.wh = self.add_weight(shape=(self.vfeat_len, self.embed_dim),
                                  initializer=self.kernel_initializer,
                                  name='{}_wh'.format(self.name))
        self.wv = self.add_weight(shape=(self.embed_dim, self.vfeat_dim),
                                  initializer=self.kernel_initializer,
                                  name='{}_wv'.format(self.name))
        self.wg = self.add_weight(shape=(self.embed_dim, seqdim),
                                  initializer=self.kernel_initializer,
                                  name='{}_wg'.format(self.name))
        super().build(input_shape)

    def call(self, x, mask=None):
        assert isinstance(x, list)
        h, v = x

        s = K.dot(self.wv, v) + K.dot(K.dot(self.wg, h), K.ones((1, self.vfeat_len)))
        z = K.dot(self.wh, K.tanh(s))
        alpha = K.softmax(z)

        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask

        weighted_avg = v * K.expand_dims(alpha)
        return weighted_avg


    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_h, shape_v = input_shape
        return shape_v

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
