import logging

from keras import Model
from keras.layers import concatenate, Dense, RepeatVector, Embedding, TimeDistributed, BatchNormalization, LSTM, Input, \
    Flatten, Convolution1D, Activation, add, multiply, GlobalAveragePooling1D, Reshape
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

    def word_attn_model(self):
        """
        Model to generate attention at each time step.
        Attention is composed of 2 parts:
        - image localized features (constant throughout time)
        - word features (output of lstm layer)

        Output is a context-weighted average of localized visual features.
        """
        word_input = Input(shape=(self.lstm_units,))
        image_input = Input(shape=(self.num_vfeats, self.vfeats_dim))

        word_x = Dense(self.num_vfeats)(word_input)
        word_emb = RepeatVector(self.num_vfeats)(word_x)

        image_conv = Convolution1D(self.num_vfeats, 1, paddings='same')(image_input)

        emb = add([word_emb, image_conv])
        emb_act = Activation('tanh')(emb)

        # TODO: check if the below is correct: intent is to generatea softmax across each dimension.
        activation = Dense(1, activation='softmax')
        attention = TimeDistributed(activation)(emb_act)
        attention = Reshape((self.num_vfeats,))(attention)  # tensor is now 1D
        attention = RepeatVector(self.num_vfeats)(attention)  # tensor is now 2D: num_vfeats x num_vfeats
        out = multiply([word_input, attention])

        model = Model(inputs=[word_input, image_input], output=out)
        return model

    def _build_model(self):
        img_input, global_img, local_img = self._image_model()
        word_input, word_model = self._word_model()

        merged = concatenate([word_model, global_img])
        seq_output = self._build_seq_output(merged)

        attn_model = self.word_attn_model()
        attn_layer = TimeDistributed(attn_model)([seq_output, local_img])

        pred_input = concatenate([seq_output, attn_layer])
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
        vg = GlobalAveragePooling1D()(x)

        f_vg = Dense(self.img_dense_dim, activation='relu')(vg)
        f_vi = Dense(self.vfeats_dim, activation='relu')(vi)

        rep_vg = RepeatVector(self.max_caption_len)(f_vg)
        rep_vi = RepeatVector(self.max_caption_len)(f_vi)
        return self.image_model.input, rep_vg, rep_vi

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
