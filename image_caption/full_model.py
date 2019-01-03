from keras import Model
from keras.applications import VGG16
from keras.layers import concatenate, Dense, RepeatVector, Embedding, TimeDistributed, BatchNormalization, LSTM, Input, \
    Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class E2eModel(object):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 text_embedding_matrix, embedding_dim,
                 text_embedding_trainable, img_dense_dim,
                 lstm_units, learning_rate,
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
        # TODO: instrument image model to be only partially trainable.
        # VGG model with no pooling to allow localization of features.
        self.image_model = VGG16(
            weights='imagenet', include_top=False,
            input_shape=img_embedding_shape
        )

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
        x = self.image_model.output
        x = Flatten()(x)
        x = Dense(self.img_dense_dim, activation='relu')(x)
        tx = RepeatVector(self.max_caption_len)(x)
        return self.image_model.input, tx

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
