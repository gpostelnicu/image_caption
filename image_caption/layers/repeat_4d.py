from keras import backend as K
from keras.engine import Layer, InputSpec


class RepeatVector4D(Layer):
    def __init__(self, n, **kwargs):
        self.n = n
        self.input_spec = [InputSpec(ndim=3)]
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        print('Received input shape: {}'.format(input_shape))
        return (input_shape[0], self.n, input_shape[1], input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 1)
        x = K.repeat_elements(x, self.n, 1)
        if mask is not None:
            x = x * mask
        return x

    def get_config(self):
        config = {
            'n': self.n
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
