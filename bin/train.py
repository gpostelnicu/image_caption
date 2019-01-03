# import argparse
import csv
import os
import pickle
from collections import defaultdict

import fire
import logging

import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from image_caption import Flickr8KSequence, SimpleModel
from image_caption.dataset import Flickr8kDataset, Flickr8kEncodedSequence, Flickr8kNextWordSequence, \
    Flickr8kImageSequence
from image_caption.full_model import E2eModel
from image_caption.image_encoder import ImageEncoder
from image_caption.models import EncoderDecoderModel, Word2VecNextWordModel, OneHotNextWordModel
from image_caption.utils import setup_logging, load_fasttext, create_embedding_matrix


def train_out_onehot(
    train_image_encodings_path,
    training_captions_path,
    test_image_encodings_path,
    test_captions_path,
    embeddings_path,
    output_prefix,
    num_epochs,
    embedding_dim=300,
    img_dense_dim=128,
    lstm_units=128,
    batch_size=64,
    num_image_versions=5,
    learning_rate=1e-5,
    dropout=0.1,
    recurrent_dropout=0.1,
    decoder_dense_dim=256,
    loss='categorical_crossentropy',
    num_dense_layers=1,
    num_lstm_layers=1):
    setup_logging()

    logging.info("Loading Flickr8K train dataset.")
    train_flkr = Flickr8kDataset(captions_path=training_captions_path)
    logging.info("Loaded train dataset. Number of samples: {}, number of steps: {}".format(
        len(train_flkr.captions), len(train_flkr)
    ))
    test_flkr = Flickr8kDataset(captions_path=test_captions_path)
    logging.info("Loaded test dataset. Number of samples: {}, number of steps: {}".format(
        len(test_flkr.captions), len(test_flkr)
    ))

    # Generate tokenizer
    logging.info("Fitting tokenizer.")
    tok = Tokenizer()
    tok.fit_on_texts(train_flkr.captions)
    tok.fit_on_texts(test_flkr.captions)
    output_path = '{}-tok.pkl'.format(output_prefix)
    logging.info('Writing tokenizer file to file {}'.format(output_path))
    pickle.dump(tok, open(output_path, 'wb'))

    embeddings = load_fasttext(embeddings_path)
    embedding_matrix = create_embedding_matrix(tok.word_index, embeddings, embedding_dim, special_tokens=[])

    train_seq = Flickr8kNextWordSequence(
        train_flkr, batch_size, train_image_encodings_path,
        tok, train_flkr.max_length, num_image_versions
    )
    test_seq = Flickr8kNextWordSequence(
        test_flkr, batch_size, test_image_encodings_path,
        tok, train_flkr.max_length, num_image_versions
    )

    model = OneHotNextWordModel(
        img_encoding_shape=(512,),
        max_caption_len=train_flkr.max_length,
        vocab_size=1 + len(tok.index_word),
        embedding_dim=embedding_dim,
        text_embedding_matrix=embedding_matrix,
        lstm_units=lstm_units,
        img_dense_dim=img_dense_dim,
        learning_rate=learning_rate,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        decoder_dense_dim=decoder_dense_dim,
        num_dense_layers=num_dense_layers, num_lstm_layers=num_lstm_layers,
        loss=loss
    )

    out_model = '{}_model.h5'.format(output_prefix)
    callbacks = [
        ModelCheckpoint(out_model, save_best_only=True),
        EarlyStopping(patience=10),
        TensorBoard()
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


def train_out_w2v(
    train_image_encodings_path,
    training_captions_path,
    test_image_encodings_path,
    test_captions_path,
    embeddings_path,
    output_prefix,
    num_epochs,
    embedding_dim=300,
    img_dense_dim=128,
    lstm_units=128,
    batch_size=64,
    num_image_versions=5,
    learning_rate=1e-5,
    dropout=0.1,
    recurrent_dropout=0.1,
    decoder_dense_dim=256,
    loss='mse',
    num_dense_layers=1):
    setup_logging()

    logging.info("Loading Flickr8K train dataset.")
    train_flkr = Flickr8kDataset(captions_path=training_captions_path)
    logging.info("Loaded train dataset. Number of samples: {}, number of steps: {}".format(
        len(train_flkr.captions), len(train_flkr)
    ))
    test_flkr = Flickr8kDataset(captions_path=test_captions_path)
    logging.info("Loaded test dataset. Number of samples: {}, number of steps: {}".format(
        len(test_flkr.captions), len(test_flkr)
    ))

    # Generate tokenizer
    tok = Tokenizer()
    tok.fit_on_texts(train_flkr.captions)
    tok.fit_on_texts(test_flkr.captions)
    output_path = '{}-tok.pkl'.format(output_prefix)
    logging.info('Writing tokenizer file to file {}'.format(output_path))
    pickle.dump(tok, open(output_path, 'wb'))

    special_tokens = ['starttoken', 'endtoken']
    embeddings = load_fasttext(embeddings_path)
    embedding_matrix = create_embedding_matrix(tok.word_index, embeddings, embedding_dim,
                                               special_tokens=special_tokens)

    train_seq = Flickr8kNextWordSequence(
        train_flkr, batch_size, train_image_encodings_path,
        tok, train_flkr.max_length, num_image_versions, embedding_matrix
    )
    test_seq = Flickr8kNextWordSequence(
        test_flkr, batch_size, test_image_encodings_path,
        tok, train_flkr.max_length, num_image_versions, embedding_matrix
    )

    model = Word2VecNextWordModel(
        img_encoding_shape=(512,),
        max_caption_len=train_flkr.max_length,
        vocab_size=1 + len(tok.index_word),
        embedding_dim=embedding_dim + len(special_tokens),
        text_embedding_matrix=embedding_matrix,
        lstm_units=lstm_units,
        img_dense_dim=img_dense_dim,
        learning_rate=learning_rate,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        decoder_dense_dim=decoder_dense_dim,
        num_dense_layers=num_dense_layers,
        loss=loss
    )

    out_model = '{}_model.h5'.format(output_prefix)
    callbacks = [
        ModelCheckpoint(out_model, save_best_only=True),
        EarlyStopping(patience=10),
        TensorBoard()
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


def train(train_image_encodings_path,
          training_captions_path,
          test_image_encodings_path,
          test_captions_path,
          embeddings_path,
          output_prefix,
          num_epochs,
          embedding_dim=300,
          img_dense_dim=128,
          lstm_units=128,
          batch_size=64,
          learning_rate=1e-5,
          num_image_versions=5,
          text_embedding_trainable=False,
          dropout=0.0,
          recurrent_dropout=0.0
          ):
    setup_logging()

    logging.info("Loading Flickr8K train dataset.")
    train_flkr = Flickr8kDataset(captions_path=training_captions_path)
    logging.info("Loaded train dataset. Number of samples: {}.".format(len(train_flkr.captions)))
    test_flkr = Flickr8kDataset(captions_path=test_captions_path)
    logging.info("Loaded test dataset. Number of samples: {}.".format(len(test_flkr.captions)))

    tok = Tokenizer()
    tok.fit_on_texts(train_flkr.captions)
    tok.fit_on_texts(test_flkr.captions)
    output_path = '{}_tok.pkl'.format(output_prefix)
    logging.info('Writing tokenizer file to file {}'.format(output_path))
    pickle.dump(tok, open(output_path, 'wb'))

    train_seq = Flickr8kEncodedSequence(
        train_flkr, batch_size, train_image_encodings_path,
        tok, train_flkr.max_length, num_image_versions
    )
    logging.info("Number of train steps: {}".format(len(train_seq)))
    test_seq = Flickr8kEncodedSequence(
        test_flkr, batch_size, test_image_encodings_path,
        tok, train_flkr.max_length, num_image_versions
    )
    logging.info("Number of test steps: {}.".format(len(test_seq)))

    special_tokens = ['starttoken', 'endtoken']
    embeddings = load_fasttext(embeddings_path)
    embedding_matrix = create_embedding_matrix(tok.word_index, embeddings, embedding_dim,
                                               special_tokens=special_tokens)

    model = SimpleModel(
        text_embedding_matrix=embedding_matrix,
        img_embedding_shape=(512,),
        max_caption_len=train_flkr.max_length,
        vocab_size=1 + len(tok.index_word),
        embedding_dim=embedding_dim + len(special_tokens),
        text_embedding_trainable=text_embedding_trainable,
        img_dense_dim=img_dense_dim,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    )

    out_model = '{}_model.h5'.format(output_prefix)
    callbacks = [
        ModelCheckpoint(out_model, save_best_only=True),
        EarlyStopping(patience=10),
        TensorBoard()
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


def train_e2e(images_dir,
              training_captions_path,
              test_captions_path,
              embeddings_path,
              output_prefix,
              num_epochs,
              embedding_dim=300,
              img_dense_dim=1024,
              lstm_units=128,
              batch_size=16,
              learning_rate=1e-5,
              text_embedding_trainable=False,
              dropout=0.0,
              recurrent_dropout=0.0
              ):
    setup_logging()

    logging.info("Loading Flickr8K train dataset.")
    train_flkr = Flickr8kDataset(captions_path=training_captions_path)
    logging.info("Loaded train dataset. Number of samples: {}.".format(len(train_flkr.captions)))
    test_flkr = Flickr8kDataset(captions_path=test_captions_path)
    logging.info("Loaded test dataset. Number of samples: {}.".format(len(test_flkr.captions)))

    tok = Tokenizer()
    tok.fit_on_texts(train_flkr.captions)
    tok.fit_on_texts(test_flkr.captions)
    output_path = '{}_tok.pkl'.format(output_prefix)
    logging.info('Writing tokenizer file to file {}'.format(output_path))
    pickle.dump(tok, open(output_path, 'wb'))

    logging.info("Setting max_len to be : {}".format(train_flkr.max_length))
    train_seq = Flickr8kImageSequence(
        train_flkr, images_dir, batch_size, tok, train_flkr.max_length
    )
    logging.info("Number of train steps: {}".format(len(train_seq)))
    test_seq = Flickr8kImageSequence(
        test_flkr, images_dir, batch_size, tok, train_flkr.max_length
    )
    logging.info("Number of test steps: {}.".format(len(test_seq)))

    special_tokens = ['starttoken', 'endtoken']
    embeddings = load_fasttext(embeddings_path)
    embedding_matrix = create_embedding_matrix(tok.word_index, embeddings, embedding_dim,
                                               special_tokens=special_tokens)

    model = E2eModel(
        img_embedding_shape=(224, 224, 3),
        text_embedding_matrix=embedding_matrix,
        max_caption_len=train_flkr.max_length,
        vocab_size=1 + len(tok.index_word),
        embedding_dim=embedding_dim + len(special_tokens),
        text_embedding_trainable=text_embedding_trainable,
        img_dense_dim=img_dense_dim,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout
    )

    out_model = '{}_model.h5'.format(output_prefix)
    callbacks = [
        ModelCheckpoint(out_model, save_best_only=True),
        EarlyStopping(patience=10),
        TensorBoard()
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


def encode_images(image_ids_path, im_dir, output_encodings, num_image_transforms=5):
    setup_logging()
    image_ids = open(image_ids_path).read().split('\n')[:-1]

    encoder = ImageEncoder(random_transform=True)

    im_encodings = {}
    for imid in image_ids:
        im_path = os.path.join(im_dir, imid)
        logging.info("Reading image {}".format(im_path))
        im = image.load_img(im_path, target_size=(224, 224))
        for i in range(num_image_transforms):
            prediction = encoder.process(im)
            im_encodings['{}-{}'.format(imid, i)] = prediction

    with open(output_encodings, 'wb') as fh:
        pickle.dump(im_encodings, fh)


def inference_lstm(im_path, model_path, tok_path, max_cap_len=39):
    """
    Perform inference using a model trained to predict LSTM.
    """
    tok = pickle.load(open(tok_path, 'rb'))
    model = load_model(model_path)
    encoder = ImageEncoder(random_transform=False)
    im_encoding = encoder.process(im_path)

    def encode_partial_cap(partial_cap, im):
        input_text = [[tok.word_index[w] for w in partial_cap if w in tok.word_index]]
        input_text = pad_sequences(input_text, maxlen=max_cap_len, padding='post')
        im = np.array([im])
        return [im, input_text]

    partial_cap = ['starttoken']
    EOS_TOKEN = 'endtoken'

    while True:
        inputs = encode_partial_cap(partial_cap, im_encoding)
        preds = model.predict(inputs)[0, len(partial_cap), :]
        next_idx = np.argmax(preds, axis=-1)
        next_word = tok.index_word[next_idx]
        if next_word == EOS_TOKEN or len(partial_cap) == 39:
            break
        partial_cap.append(next_word)

    print(' '.join(partial_cap[1:]))  # skip starttoken.


def inference_word(im_path, model_path, tok_path, max_cap_len=39):
    """
    Perform inference using a model trained by word, as opposed to LSTM.
    """
    tok = pickle.load(open(tok_path, 'rb'))
    model = load_model(model_path)
    encoder = ImageEncoder(random_transform=False)
    im_encoding = encoder.process(im_path)

    def encode_partial_cap(partial_cap, im):
        input_text = [[tok.word_index[w] for w in partial_cap if w in tok.word_index]]
        input_text = pad_sequences(input_text, maxlen=max_cap_len, padding='post')
        im = np.array([im])
        return [im, input_text]

    partial_cap = ['starttoken']
    EOS_TOKEN = 'endtoken'

    while True:
        inputs = encode_partial_cap(partial_cap, im_encoding)
        preds = model.predict(inputs)
        next_idx = np.argmax(preds, axis=-1)[0]
        next_word = tok.index_word[next_idx]
        if next_word == EOS_TOKEN or len(partial_cap) == 39:
            break
        partial_cap.append(next_word)

    print('Predicted: {}'.format(' '.join(partial_cap)))


def encode_text(image_captions_path, imids_path, output_path):
    captions = defaultdict(list)
    with open(image_captions_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            imid = row[0]
            imid = imid[:len(imid) - 2]  # strip annotation number
            caption = row[1]
            captions[imid].append(caption)

    imids = open(imids_path).read().split('\n')[:-1]
    with open(output_path, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for imid in imids:
            im_captions = captions[imid]
            for caption in im_captions:
                writer.writerow([
                    imid,
                    'starttoken {} endtoken'.format(caption)
                ])


if __name__ == '__main__':
    fire.Fire()
