import logging

from keras.models import Model
from keras.layers import concatenate, Dense, RepeatVector, Embedding, TimeDistributed, BatchNormalization, LSTM, Input, \
    Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop


class E2eModel(object):
    def __init__(self, img_embedding_shape, max_caption_len, vocab_size,
                 text_embedding_matrix, embedding_dim,
                 text_embedding_trainable, img_dense_dim,
                 lstm_units, learning_rate,
                 dropout, recurrent_dropout,
                 image_layers_to_unfreeze,
                 cnn_model, image_pooling, mask_zeros,
                 num_lstm_layers=1,
                 additional_dense_layer_dim=None):
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
        self.num_lstm_layers = num_lstm_layers
        self.additional_dense_layer_dim = additional_dense_layer_dim

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
        word_input, word_model = self._word_model(self.max_caption_len)

        merged = concatenate([word_model, img_model])
        seq_output = self._build_seq_output(merged)

        model = Model(inputs=[img_input, word_input],
                      outputs=seq_output)
        model.compile(optimizer=RMSprop(lr=self.learning_rate, clipnorm=1.0),
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

    def _word_model(self, seq_len):
        word_input = Input(shape=(seq_len,), dtype='int32', name='text_input')
        if self.text_embedding_matrix is not None:
            embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.text_embedding_matrix],
                                  input_length=seq_len, trainable=self.text_embedding_trainable,
                                  mask_zero=self.mask_zeros)(word_input)
        else:
            logging.info("Empty embeddings weights")
            embedding = Embedding(self.vocab_size, self.embedding_dim,
                                  input_length=seq_len, trainable=True,
                                  mask_zero=self.mask_zeros)(word_input)
        return word_input, embedding

    def _build_seq_output(self, sequence_input):
        x = TimeDistributed(BatchNormalization(axis=-1))(sequence_input)
        for _ in range(self.num_lstm_layers):
            x = LSTM(units=self.lstm_units, return_sequences=True,
                     dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(x)
        if self.additional_dense_layer_dim:
            x = TimeDistributed(Dense(self.additional_dense_layer_dim, activation='relu'))(x)
        time_dist_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)
        return time_dist_dense


class ImageFirstE2EModel(E2eModel):
    def __init__(self, cnn_dropout, text_dropout, **kwargs):
        self.cnn_dropout = cnn_dropout
        self.text_dropout = text_dropout
        super().__init__(**kwargs)  # Calls _build_model

    def _build_model(self):
        img_input, img_model = self._image_model()
        transformed_img = Dense(self.embedding_dim)(img_model)
        transformed_img = Dropout(self.cnn_dropout)(transformed_img)
        tt_img = RepeatVector(1)(transformed_img)

        word_input, word_model = self._word_model(self.max_caption_len - 1)
        if self.text_dropout > 0:
            word_model = Dropout(self.text_dropout)(word_model)

        merged = concatenate([tt_img, word_model], axis=-2)  # Concatenation adds one time step.
        seq_output = self._build_seq_output(merged)

        model = Model(inputs=[img_input, word_input],
                      outputs=seq_output)
        model.compile(optimizer=RMSprop(lr=self.learning_rate, clipnorm=1.0),
                      loss=categorical_crossentropy, sample_weight_mode='temporal')
        model.summary()
        return model


