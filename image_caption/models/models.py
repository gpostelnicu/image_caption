import logging

from keras import Model
from keras.layers import concatenate, Input, Dense, RepeatVector, Embedding, BatchNormalization, Bidirectional, LSTM, \
    TimeDistributed, Add, Concatenate, Multiply, add, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class OneHotNextWordModel(object):
    def __init__(self, cnn_model, max_caption_len, vocab_size,
                 embedding_dim, text_embedding_matrix, lstm_units, img_embedding_shape=(224, 224, 3),
                 img_dense_dim=256, decoder_dense_dim=256, learning_rate=1e-4,
                 dropout=0.0, recurrent_dropout=0.0, num_dense_layers=1,
                 loss='categorical_crossentropy', num_lstm_layers=1, image_pooling='avg'):
        self.max_caption_len = max_caption_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.text_embedding_matrix = text_embedding_matrix
        self.img_dense_dim = img_dense_dim
        self.lstm_units = lstm_units
        self.decoder_dense_dim = decoder_dense_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.num_dense_layers = num_dense_layers
        self.loss = loss
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=1.0)
        self.num_lstm_layers = num_lstm_layers
        self.image_pooling = image_pooling

        self.image_model = cnn_model(
            weights='imagenet', include_top=False,
            input_shape=img_embedding_shape,
            pooling=self.image_pooling
        )
        for layer in self.image_model.layers:
            layer.trainable = False

        self.keras_model = self._build_model()
        self.keras_model.summary()

    def _image_model(self):
        x = self.image_model.output
        if self.image_pooling is None:
            logging.info("Adding image flattening.")
            x = Flatten()(x)
        if self.img_dense_dim > 0:
            logging.info("Adding image dense layer with units: {}".format(self.img_dense_dim))
            x = Dense(self.img_dense_dim, activation='relu')(x)
        return self.image_model.input, x

    def _build_model(self):
        img_input, img_model = self._image_model()
        full_image = Dense(self.lstm_units, activation='relu', name='image_feature')(img_model)
        full_image = BatchNormalization()(full_image)

        text_input = Input(shape=(self.max_caption_len,), name='text_input')
        full_text = Embedding(self.vocab_size, self.embedding_dim, weights=[self.text_embedding_matrix],
                              input_length=self.max_caption_len, mask_zero=True, trainable=False)(text_input)
        for i in range(self.num_lstm_layers):
            full_text = LSTM(self.lstm_units, name='text_feature_{}'.format(i),
                             dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                             return_sequences=(i + 1 < self.num_lstm_layers))(full_text)
        full_text = BatchNormalization()(full_text)

        encoded = add([full_text, full_image])

        decoder = encoded
        for _ in range(self.num_dense_layers):
            decoder = Dense(self.decoder_dense_dim, activation='relu')(decoder)
        output = self._build_final_layer(decoder)

        model = Model(inputs=[img_input, text_input], outputs=output)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def _build_final_layer(self, decoded):
        return Dense(self.vocab_size, activation='softmax')(decoded)


class Word2VecNextWordModel(OneHotNextWordModel):
    def __init__(self, img_encoding_shape, max_caption_len, vocab_size,
                 embedding_dim, text_embedding_matrix, lstm_units,
                 img_dense_dim=256, decoder_dense_dim=256, learning_rate=1e-4,
                 dropout=0.0, recurrent_dropout=0.0, num_dense_layers=1,
                 loss='cosine', optimizer='adam'):
        super().__init__(img_encoding_shape, max_caption_len, vocab_size,
                         embedding_dim, text_embedding_matrix, lstm_units,
                         img_dense_dim, decoder_dense_dim, learning_rate,
                         dropout, recurrent_dropout, num_dense_layers,
                         loss, optimizer)

    def _build_final_layer(self, decoded):
        return Dense(self.embedding_dim, activation='tanh')(decoded)


class EncoderDecoderModel(object):
    def __init__(self, img_encoding_shape, max_caption_len, vocab_size,
                 embedding_dim, text_embedding_matrix, lstm_units,
                 img_dense_dim=256, decoder_dense_dim=256, learning_rate=1e-4,
                 dropout=0.0, recurrent_dropout=0.0, num_dense_layers=1, loss='categorical_crossentropy'):
        self.img_encoding_shape = img_encoding_shape
        self.max_caption_len = max_caption_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.text_embedding_matrix = text_embedding_matrix
        self.img_dense_dim = img_dense_dim
        self.lstm_units = lstm_units
        self.decoder_dense_dim = decoder_dense_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.num_dense_layers = num_dense_layers
        self.loss = loss

        self.keras_model = self._build_model()
        self.keras_model.summary()

    def _build_model(self):
        image_input = Input(shape=self.img_encoding_shape, name='image_input')
        full_image = Dense(self.lstm_units, activation='relu', name='image_feature')(image_input)
        full_image = BatchNormalization()(full_image)

        text_input = Input(shape=(self.max_caption_len,), name='text_input')
        full_text = Embedding(self.vocab_size, self.embedding_dim, weights=[self.text_embedding_matrix],
                              input_length=self.max_caption_len, mask_zero=True, trainable=False)(text_input)
        full_text = LSTM(self.lstm_units, name='text_feature',
                         dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(full_text)
        full_text = BatchNormalization()(full_text)

        encoded = Add()([full_text, full_image])

        decoder = encoded
        for _ in range(self.num_dense_layers):
            decoder = Dense(self.decoder_dense_dim, activation='relu')(decoder)
        output = Dense(self.embedding_dim, activation='tanh')(decoder)

        model = Model(inputs=[image_input, text_input], outputs=output)
        model.compile(loss=self.loss, optimizer='adam')

        return model


class SimpleModel(object):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 text_embedding_matrix, embedding_dim, text_embedding_trainable,
                 img_dense_dim, lstm_units, learning_rate,
                 dropout, recurrent_dropout):
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

        self.keras_model = self._build_model()


    def _build_model(self):
        img_input, img_model = self._image_model()
        word_input, word_model = self._word_model()

        merged = concatenate([word_model, img_model])
        seq_output = self._build_seq_output(merged)

        model = Model(inputs=[img_input, word_input],
                      outputs=seq_output)
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0),
                      loss=categorical_crossentropy)
        model.summary()
        return model

    def _image_model(self):
        image_input = Input(shape=self.img_embedding_shape, name='image_input')
        x = Dense(self.img_dense_dim, activation='relu')(image_input)
        tx = RepeatVector(self.max_caption_len)(x)
        return image_input, tx

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
        time_dist_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)
        return time_dist_dense
