#import argparse
import csv
import os
import pickle
from collections import defaultdict

import fire
import logging

import numpy as np
from keras.applications import VGG16

from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image

from image_caption import Flickr8KSequence, SimpleModel
from image_caption.utils import setup_logging


def train(train_image_encodings_path,
          training_captions_path,
          test_image_encodings_path,
          test_captions_path,
          output_path,
          num_epochs,
          embedding_dim=256,
          img_dense_dim=128,
          lstm_units=128,
          batch_size=64
          ):
    setup_logging()

    logging.info("Loading Flickr8K dataset.")
    train_flkr = Flickr8KSequence(batch_size,
                                  encodings_path=train_image_encodings_path,
                                  captions_path=training_captions_path)
    logging.info("Loaded dataset. Number of samples: {}, number of steps: {}".format(
        len(train_flkr.captions), len(train_flkr)
    ))
    test_flkr = Flickr8KSequence(
        batch_size,
        encodings_path=test_image_encodings_path,
        captions_path=test_captions_path,
        max_length=train_flkr.max_length,
        index_word=train_flkr.index_word
    )
    logging.info("Loaded test dataset. Number of samples: {}, number of steps: {}".format(
        len(test_flkr.captions), len(test_flkr)
    ))


    model = SimpleModel(
        img_embedding_shape=(512,),
        max_caption_len=train_flkr.max_length,
        vocab_size=train_flkr.max_vocab_index,
        embedding_dim=embedding_dim,
        img_dense_dim=img_dense_dim,
        lstm_units=lstm_units
    )

    callbacks = [
        ModelCheckpoint(output_path, save_best_only=True),
        EarlyStopping(patience=10)
    ]
    model.keras_model.fit_generator(
        train_flkr,
        steps_per_epoch=len(train_flkr),
        validation_data=test_flkr,
        validation_steps=len(test_flkr),
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks
    )


def encode_images(image_ids_path, im_dir, output_encodings):
    setup_logging()
    image_ids = open(image_ids_path).read().split('\n')[:-1]

    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    im_encodings = {}
    for imid in image_ids:
        im_path = os.path.join(im_dir, imid)
        logging.info("Reading image {}".format(im_path))
        im = image.load_img(im_path, target_size=(224, 224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        norm_im = np.asarray(x)
        prediction = model.predict(norm_im)
        prediction = np.reshape(prediction, prediction.shape[1])

        im_encodings[imid] = prediction

    with open(output_encodings, 'wb') as fh:
        pickle.dump(im_encodings, fh)


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
                    '<start> {} <end>'.format(caption)
                ])


if __name__ == '__main__':
    fire.Fire()
