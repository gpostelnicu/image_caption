import logging
import pickle

import fire
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from keras.preprocessing.text import Tokenizer

from image_caption.architectures import CNN_ARCHITECTURES
from image_caption.dataset import Flickr8kDataset, Flickr8kImageSequence
from image_caption.models import E2eModel, ImageFirstE2EModel
from image_caption.utils import setup_logging, load_fasttext, create_embedding_matrix, filter_tokenizer


class E2ETrainer(object):
    def __init__(self,
                 img_dense_dim=1024,
                 train_patience=10,
                 learning_rate=1e-4,
                 replace_words_ratio=0.0,
                 use_sample_weights=False,
                 mask_zeros=True,
                 lstm_units=512,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 cnn_architecture='resnet50',
                 image_layers_to_unfreeze=4,
                 pooling=None,
                 lr_epochs=5,
                 lr_factor=2.,
                 num_lstm_layers=1
                 ):
        self.img_dense_dim = img_dense_dim
        self.train_patience = train_patience
        self.learning_rate = learning_rate
        self.replace_words_ratio = replace_words_ratio

        self.use_sample_weights = use_sample_weights
        self.mask_zeros = mask_zeros

        self.lstm_units = lstm_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.cnn_architecture = cnn_architecture
        self.image_layers_to_unfreeze = image_layers_to_unfreeze
        self.pooling = pooling
        self.lr_epochs = lr_epochs
        self.lr_factor = lr_factor

        self.num_lstm_layers = num_lstm_layers

        self.cnn_arch = CNN_ARCHITECTURES[self.cnn_architecture]
        self.captions_start_idx = 0

        setup_logging()

    def prepare_data(self,
                     images_dir,
                     captions_path,
                     train_imids_path,
                     test_imids_path,
                     checkpoint_prefix,
                     output_prefix,
                     batch_size,
                     special_tokens
                     ):
        logging.info("Loading Flickr8K train dataset.")
        train_flkr = Flickr8kDataset(captions_path=captions_path, imids_path=train_imids_path)
        logging.info("Loaded train dataset. Number of samples: {}.".format(len(train_flkr)))
        test_flkr = Flickr8kDataset(captions_path=captions_path, imids_path=test_imids_path)
        logging.info("Loaded test dataset. Number of samples: {}.".format(len(test_flkr)))

        if checkpoint_prefix is not None:
            tok_path = '{}_tok.pkl'.format(checkpoint_prefix)
            logging.info("Loading tokenizer from file: {}".format(tok_path))
            tok = pickle.load(open(tok_path, 'rb'))
        else:
            logging.info("Generating tokenizer.")
            tok = Tokenizer(oov_token='unknowntoken')
            tok.fit_on_texts(cap for _, cap in train_flkr)
            self.add_words(tok, special_tokens)

            # Remove words that appear in less than 5 documents.
            filter_tokenizer(tok, 5)
            logging.info('Words in tokenizer: {}'.format(len(tok.word_index)))

            output_path = '{}_tok.pkl'.format(output_prefix)
            logging.info('Writing tokenizer file to file {}'.format(output_path))
            pickle.dump(tok, open(output_path, 'wb'))

        max_num_words = max(len(tok.texts_to_sequences(cap)) for _, cap in train_flkr)
        logging.info("Setting max_len to be : {}".format(max_num_words))
        train_seq = Flickr8kImageSequence(
            train_flkr, images_dir, batch_size, tok,
            max_length=max_num_words,
            image_preprocess_fn=self.cnn_arch.preprocess_fn, random_image_transform=True,
            replace_words_ratio=self.replace_words_ratio, output_weights=self.use_sample_weights
        )
        logging.info("Number of train steps: {}".format(len(train_seq)))
        test_seq = Flickr8kImageSequence(
            test_flkr, images_dir, batch_size, tok, max_length=max_num_words,
            image_preprocess_fn=self.cnn_arch.preprocess_fn, output_weights=self.use_sample_weights
        )
        logging.info("Number of test steps: {}.".format(len(test_seq)))

        return tok, train_seq, test_seq

    def do_train(self, model, train_seq, test_seq, output_prefix, num_epochs):
        out_model = '{}_model.h5'.format(output_prefix)

        def sched(epoch, lr):
            if epoch % self.lr_epochs == self.lr_epochs - 1:
                return lr / self.lr_factor
            return lr

        callbacks = [
            ModelCheckpoint(out_model, save_best_only=True),
            EarlyStopping(patience=self.train_patience),
            LearningRateScheduler(
                schedule=sched, verbose=1),
            TensorBoard(log_dir='{}_logs'.format(output_prefix))
        ]

        model.keras_model.fit_generator(
            train_seq,
            steps_per_epoch=len(train_seq),
            validation_data=test_seq,
            validation_steps=len(test_seq),
            epochs=num_epochs,
            verbose=1,
            callbacks=callbacks
        )

    def add_words(self, tok, words):
        idx = max(tok.index_word.keys())
        idx += 1

        for word in words:
            assert word not in tok.word_index, word
            tok.index_word[idx] = word
            tok.word_index[word] = idx
            idx += 1

    def train(self,
              images_dir,
              embeddings_path,
              embedding_dim,
              captions_path,
              train_imids_path,
              test_imids_path,
              output_prefix,
              num_epochs,
              batch_size,
              checkpoint_prefix=None
              ):
        special_tokens = ['starttoken', 'endtoken']
        tok, train_seq, test_seq = self.prepare_data(
            images_dir=images_dir, captions_path=captions_path, train_imids_path=train_imids_path,
            test_imids_path=test_imids_path, checkpoint_prefix=checkpoint_prefix,
            output_prefix=output_prefix, batch_size=batch_size, special_tokens=special_tokens
        )

        embeddings = load_fasttext(embeddings_path)
        embedding_matrix = create_embedding_matrix(tok.word_index, embeddings, embedding_dim,
                                                   special_tokens=special_tokens)

        model = E2eModel(
            img_embedding_shape=(224, 224, 3),
            text_embedding_matrix=embedding_matrix,
            max_caption_len=train_seq.max_length,
            vocab_size=1 + len(tok.index_word),
            embedding_dim=embedding_dim + len(special_tokens),
            text_embedding_trainable=False,
            img_dense_dim=self.img_dense_dim,
            lstm_units=self.lstm_units,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            image_layers_to_unfreeze=self.image_layers_to_unfreeze,
            cnn_model=self.cnn_arch.model,
            image_pooling=self.pooling,
            mask_zeros=self.mask_zeros
        )
        if checkpoint_prefix is not None:
            model_path = '{}_model.h5'.format(checkpoint_prefix)
            logging.info("Loading model from checkpoint {}".format(model_path))
            model.keras_model.load_weights(model_path)

        self.do_train(model, train_seq, test_seq, output_prefix, num_epochs)


class ImageFirstE2ETrainer(E2ETrainer):
    def __init__(self,
                 img_dense_dim=1024,
                 train_patience=10,
                 learning_rate=1e-4,
                 use_sample_weights=False,
                 mask_zeros=True,
                 lstm_units=512,
                 dropout=0.0,
                 replace_words_ratio=0.0,
                 recurrent_dropout=0.0,
                 cnn_architecture='resnet50',
                 image_layers_to_unfreeze=4,
                 pooling=None,
                 lr_epochs=5,
                 lr_factor=2.,
                 num_lstm_layers=1,
                 additional_dense_layer_dim=None,
                 cnn_dropout=0.0,
                 text_dropout=0.0,
                 text_embedding_trainable=False
                 ):
        super().__init__(img_dense_dim=img_dense_dim,
                         train_patience=train_patience,
                         learning_rate=learning_rate,
                         use_sample_weights=use_sample_weights,
                         mask_zeros=mask_zeros,
                         lstm_units=lstm_units,
                         dropout=dropout,
                         replace_words_ratio=replace_words_ratio,
                         recurrent_dropout=recurrent_dropout,
                         cnn_architecture=cnn_architecture,
                         image_layers_to_unfreeze=image_layers_to_unfreeze,
                         pooling=pooling, lr_epochs=lr_epochs, lr_factor=lr_factor,
                         num_lstm_layers=num_lstm_layers
                         )
        self.captions_start_idx = 1  # Override base class variable to skip <start> token.
        self.additional_dense_layer_dim = additional_dense_layer_dim
        self.cnn_dropout = cnn_dropout
        self.text_dropout = text_dropout
        self.text_embedding_trainable = text_embedding_trainable

    def train(self,
              images_dir,
              embedding_dim,
              captions_path,
              train_imids_path,
              test_imids_path,
              output_prefix,
              num_epochs,
              batch_size,
              embeddings_path=None,
              checkpoint_prefix=None
              ):
        special_tokens = ['starttoken', 'endtoken', 'unknowntoken']
        tok, train_seq, test_seq = self.prepare_data(
            images_dir=images_dir, captions_path=captions_path, train_imids_path=train_imids_path,
            test_imids_path=test_imids_path, checkpoint_prefix=checkpoint_prefix, output_prefix=output_prefix,
            batch_size=batch_size, special_tokens=special_tokens
        )

        embedding_matrix = None
        if embeddings_path:
            embeddings = load_fasttext(embeddings_path)
            embedding_matrix = create_embedding_matrix(tok.word_index, embeddings, embedding_dim,
                                                       special_tokens=special_tokens)

        model = ImageFirstE2EModel(
            cnn_dropout=self.cnn_dropout,
            text_dropout=self.text_dropout,
            img_embedding_shape=(224, 224, 3),
            text_embedding_matrix=embedding_matrix,
            max_caption_len=train_seq.max_length,
            vocab_size=1 + len(tok.index_word),
            embedding_dim=embedding_dim + len(special_tokens),
            text_embedding_trainable=self.text_embedding_trainable,
            img_dense_dim=self.img_dense_dim,
            lstm_units=self.lstm_units,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            image_layers_to_unfreeze=self.image_layers_to_unfreeze,
            cnn_model=self.cnn_arch.model,
            image_pooling=self.pooling,
            mask_zeros=self.mask_zeros,
            additional_dense_layer_dim=self.additional_dense_layer_dim,
            num_lstm_layers=self.num_lstm_layers
        )
        if checkpoint_prefix is not None:
            model_path = '{}_model.h5'.format(checkpoint_prefix)
            logging.info("Loading model from checkpoint {}".format(model_path))
            model.keras_model.load_weights(model_path)

        self.do_train(model, train_seq, test_seq, output_prefix, num_epochs)
