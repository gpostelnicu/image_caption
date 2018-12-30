from keras import Model
from keras.layers import concatenate, Input, Dense, RepeatVector, Embedding, BatchNormalization, Bidirectional, LSTM, \
    TimeDistributed, Add
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class EncoderDecoderModel(object):
    def __init__(self, img_encoding_shape, max_caption_len, vocab_size,
                 embedding_dim, lstm_units, img_dense_dim=256, decoder_dense_dim=256, learning_rate=1e-4):
        self.img_encoding_shape = img_encoding_shape
        self.max_caption_len = max_caption_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.img_dense_dim = img_dense_dim
        self.lstm_units = lstm_units
        self.decoder_dense_dim = decoder_dense_dim
        self.learning_rate = learning_rate

        self.keras_model = self._build_model()
        self.keras_model.summary()

    def _build_model(self):
        image_input = Input(shape=self.img_encoding_shape, name='image_input')
        full_image = Dense(self.lstm_units, activation='relu', name='image_feature')(image_input)

        text_input = Input(shape=(self.max_caption_len,), name='text_input')
        full_text = Embedding(self.vocab_size, self.embedding_dim,
                              input_length=self.max_caption_len, mask_zero=True)(text_input)
        full_text = LSTM(self.lstm_units, name='text_feature')(full_text)

        encoded = Add()([full_text, full_image])

        decoder = Dense(self.decoder_dense_dim, activation='relu')(encoded)
        output = Dense(self.vocab_size, activation='softmax')(decoder)

        model = Model(inputs=[image_input, text_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model


class SimpleModel(object):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 embedding_dim, img_dense_dim, lstm_units, learning_rate):
        self.img_embedding_shape = img_embedding_shape
        self.max_caption_len = max_caption_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.img_dense_dim = img_dense_dim
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate

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
        image_input = Input(shape=self.img_embedding_shape)
        x = Dense(self.img_dense_dim, activation='relu')(image_input)
        tx = RepeatVector(self.max_caption_len)(x)
        return image_input, tx

    def _word_model(self):
        word_input = Input(shape=(self.max_caption_len,))
        embedding = Embedding(self.vocab_size, self.embedding_dim,
                              input_length=self.max_caption_len)(word_input)
        return word_input, embedding

    def _build_seq_output(self, sequence_input):
            x = BatchNormalization(axis=-1)(sequence_input)
            x = Bidirectional(LSTM(units=self.lstm_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
            time_dist_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)
            return time_dist_dense


