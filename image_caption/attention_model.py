import logging

from keras import backend as K
from keras import Model
from keras.layers import concatenate, Dense, RepeatVector, Embedding, TimeDistributed, BatchNormalization, LSTM, Input, \
    Flatten, Convolution1D, Activation, add, multiply, GlobalAveragePooling1D, Reshape, Permute, Lambda
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from image_caption.layers.attention import AttentionWeightedAverage
from image_caption.layers.repeat_4d import RepeatVector4D


class AttentionModel(object):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 text_embedding_matrix, embedding_dim,
                 text_embedding_trainable, img_dense_dim,
                 lstm_units, learning_rate,
                 dropout, recurrent_dropout,
                 image_layers_to_unfreeze,
                 cnn_model, num_vfeats, vfeats_dim):
        self.img_embedding_shape = img_embedding_shape
        self.max_caption_len = max_caption_len
        self.vocab_size = vocab_size
        self.text_embedding_matrix = text_embedding_matrix
        self.embedding_dim = embedding_dim
        self.text_embedding_trainable = text_embedding_trainable
        self.img_dense_dim = img_dense_dim
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.num_vfeats = num_vfeats
        self.vfeats_dim = vfeats_dim
        self.attn_embed_dim = 100
        self.mask_zero = False

        # TODO: instrument image model to be only partially trainable.
        self.image_model = cnn_model(
            weights='imagenet', include_top=False,
            input_shape=img_embedding_shape,
            pooling=None
        )
        logging.info("Freezing all layers except last {}".format
                     (image_layers_to_unfreeze))
        for layer in self.image_model.layers[:-image_layers_to_unfreeze]:
            layer.trainable = False

        self.keras_model = self._build_model()

    def _attention(self, vi, h):
        vi = K.print_tensor(vi, message='vi = ')
        h = K.print_tensor(h, message='h = ')
        
        z_vi = TimeDistributed(Convolution1D(self.attn_embed_dim, 1, padding='same'))(vi)

        z_h = TimeDistributed(Dense(self.attn_embed_dim))(h)
        z_h = TimeDistributed(RepeatVector(self.num_vfeats))(z_h)

        tz_vi = RepeatVector4D(self.max_caption_len)(z_vi)
        s = add([tz_vi, z_h])
        alpha = Activation('tanh')(s)
        alpha = TimeDistributed(TimeDistributed(Dense(1)))(alpha)
        alpha = TimeDistributed(Activation('softmax'))(alpha)

        t_alpha = TimeDistributed(RepeatVector(self.vfeats_dim))(alpha)
        weighted = multiply([vi, t_alpha])
        w_avg = Lambda(lambda x: K.sum(x, axis=1))(weighted)
        return w_avg

    def _build_model(self):
        img_input, global_img, local_img = self._image_model()
        word_input, word_model = self._word_model()

        merged = concatenate([word_model, global_img])
        seq_output = self._build_seq_output(merged)

        vi = RepeatVector4D(self.max_caption_len)(local_img)
        attn_local_img = self._attention(vi, seq_output)

        pred_input = concatenate([seq_output, attn_local_img])
        output = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(pred_input)

        model = Model(inputs=[img_input, word_input],
                      outputs=output)
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0),
                      loss=categorical_crossentropy)
        model.summary()
        return model

    def _image_model(self):
        x = self.image_model.output
        vi = Reshape((self.num_vfeats, self.vfeats_dim))(x)
        vg = GlobalAveragePooling1D()(vi)

        f_vg = Dense(self.img_dense_dim, activation='relu')(vg)

        rep_vg = RepeatVector(self.max_caption_len)(f_vg)
        vi = Convolution1D(self.vfeats_dim, 1, padding='same', activation='relu')(vi)
        return self.image_model.input, rep_vg, vi

    def _word_model(self):
        word_input = Input(shape=(self.max_caption_len,), dtype='int32', name='text_input')
        embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.text_embedding_matrix],
                              input_length=self.max_caption_len, trainable=self.text_embedding_trainable,
                              mask_zero=self.mask_zero)(word_input)
        return word_input, embedding

    def _build_seq_output(self, sequence_input):
        x = TimeDistributed(BatchNormalization(axis=-1))(sequence_input)
        x = LSTM(units=self.lstm_units, return_sequences=True,
                 dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(x)
        return x
