import logging

from keras.models import Model
from keras.layers import concatenate, Dense, RepeatVector, Embedding, TimeDistributed, BatchNormalization, LSTM, Input, \
    Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class E2eModel(object):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 text_embedding_matrix, embedding_dim,
                 text_embedding_trainable, img_dense_dim,
                 lstm_units, learning_rate,
                 dropout, recurrent_dropout,
                 image_layers_to_unfreeze,
                 cnn_model, image_pooling, mask_zeros):
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
        self.image_pooling = image_pooling
        self.mask_zeros = mask_zeros

        self.image_model = cnn_model(
            weights='imagenet', include_top=False,
            input_shape=img_embedding_shape,
            pooling=self.image_pooling
        )
        logging.info("Freezing all layers except last {}".format
                     (image_layers_to_unfreeze))
        for layer in self.image_model.layers[:-image_layers_to_unfreeze]:
            layer.trainable = False

        self.keras_model = self._build_model()

    def _build_model(self):
        img_input, img_model = self._image_model()
        img_model = RepeatVector(self.max_caption_len)(img_model)
        word_input, word_model = self._word_model()

        merged = concatenate([word_model, img_model])
        seq_output = self._build_seq_output(merged)

        model = Model(inputs=[img_input, word_input],
                      outputs=seq_output)
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0),
                      loss=categorical_crossentropy, sample_weight_mode='temporal')
        model.summary()
        return model

    def _image_model(self):
        x = self.image_model.output
        if self.image_pooling is None:
            logging.info("Adding image flattening.")
            x = Flatten()(x)
        if self.img_dense_dim > 0:
            logging.info("Adding image dense layer with units: {}".format(self.img_dense_dim))
            x = Dense(self.img_dense_dim, activation='relu')(x)
        return self.image_model.input, x

    def _word_model(self):
        word_input = Input(shape=(self.max_caption_len,), dtype='int32', name='text_input')
        embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.text_embedding_matrix],
                              input_length=self.max_caption_len, trainable=self.text_embedding_trainable,
                              mask_zero=self.mask_zeros)(word_input)
        return word_input, embedding

    def _build_seq_output(self, sequence_input):
        x = TimeDistributed(BatchNormalization(axis=-1))(sequence_input)
        x = LSTM(units=self.lstm_units, return_sequences=True,
                 dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(x)
        time_dist_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)
        return time_dist_dense


class ImageFirstE2EModel(E2eModel):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 text_embedding_matrix, embedding_dim,
                 text_embedding_trainable, img_dense_dim,
                 lstm_units, learning_rate,
                 dropout, recurrent_dropout,
                 image_layers_to_unfreeze,
                 cnn_model, image_pooling, mask_zeros):
        super().__init__(img_embedding_shape, max_caption_len, vocab_size,
                         text_embedding_matrix, embedding_dim,
                         text_embedding_trainable, img_dense_dim,
                         lstm_units, learning_rate,
                         dropout, recurrent_dropout,
                         image_layers_to_unfreeze,
                         cnn_model, image_pooling, mask_zeros)

    def _build_model(self):
        img_input, img_model = self._image_model()
        transformed_img = Dense(self.embedding_dim)(img_model)
        tt_img = RepeatVector(1)(transformed_img)
        word_input, word_model = self._word_model()

        merged = concatenate([tt_img, word_model], axis=-2)  # Concatenation adds one time step.
        seq_output = self._build_seq_output(merged)

        model = Model(inputs=[img_input, word_input],
                      outputs=seq_output)
        model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1.0),
                      loss=categorical_crossentropy, sample_weight_mode='temporal')
        model.summary()
        return model

    def _word_model(self):
        word_input = Input(shape=(self.max_caption_len - 1,), dtype='int32', name='text_input')
        embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.text_embedding_matrix],
                              input_length=self.max_caption_len - 1, trainable=self.text_embedding_trainable,
                              mask_zero=self.mask_zeros)(word_input)
        return word_input, embedding


