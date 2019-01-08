import logging

from keras import backend as K
from keras import Model
from keras.layers import concatenate, Dense, RepeatVector, Embedding, TimeDistributed, BatchNormalization, LSTM, Input, \
    Flatten, Convolution1D, Activation, add, multiply, GlobalAveragePooling1D, Reshape, Permute, Lambda
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


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

    def word_attn_model(self, h, vi):
        """
        linear used as input to final classifier, embedded used to compute attention
        """

        h_out_linear = TimeDistributed(Dense(self.attn_embed_dim, 1, padding='same'))(h)
        z_h_embed = TimeDistributed(RepeatVector(self.num_vfeats))(h_out_linear)

        z_v_linear = TimeDistributed(RepeatVector(self.max_caption_len))(vi)
        vi_embed = Convolution1D(self.attn_embed_dim, 1, padding='same')(vi)
        z_v_embed = TimeDistributed(RepeatVector(self.max_caption_len))(vi_embed)

        z_v_linear = Permute((2, 1, 3))(z_v_linear)
        z_v_embed = Permute((2, 1, 3))(z_v_embed)

        z = add([z_h_embed, z_v_embed])
        z = TimeDistributed(Activation('tanh'))(z)
        att = TimeDistributed(Convolution1D(1, 1, padding='same'))(z)

        att = Reshape((self.max_caption_len, self.num_vfeats))(att)
        att = TimeDistributed(Activation('softmax'))(att)
        att = TimeDistributed(RepeatVector(self.vfeats_dim))(att)
        att = Permute((1, 3, 2))(att)

        ctx = multiply([att, z_v_linear])
        # TODO: Does the lambda need special help to support masking? Does it matter?
        sum_layer = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(self.attn_embed_dim,))
        out = TimeDistributed(sum_layer)(ctx)
        return out


    def _build_model(self):
        img_input, global_img, local_img = self._image_model()
        word_input, word_model = self._word_model()

        merged = concatenate([word_model, global_img])
        seq_output = self._build_seq_output(merged)

        attn_local_img = self.word_attn_model(seq_output, local_img)

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
                              mask_zero=True)(word_input)
        return word_input, embedding

    def _build_seq_output(self, sequence_input):
        x = TimeDistributed(BatchNormalization(axis=-1))(sequence_input)
        x = LSTM(units=self.lstm_units, return_sequences=True,
                 dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(x)
        return x
