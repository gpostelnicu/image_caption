import csv
import logging
import pickle
import random

import numpy as np

from keras.utils import Sequence
from keras_preprocessing import sequence


class Flickr8KSequence(Sequence):
    def __init__(self, batch_size, encodings_path,
                 captions_path, max_length=None, index_word=None):
        self.batch_size = batch_size
        self.encodings = pickle.load(open(encodings_path, 'rb'))
        self.captions, self.image_ids = self._load_captions(captions_path)
        if max_length is None:
            self.max_length = max(len(caption) for caption in self.captions)
        else:
            self.max_length = max_length
        if index_word is None:
            logging.info("Generating index.")
            self.index_word = self._generate_index_word(self.captions)
        else:
            logging.info("Loading index from dictionary.")
            self.index_word = index_word
        self.word_index = self._generate_word_index(self.index_word)
        self.max_vocab_index = max(self.index_word.keys()) + 1

        # Indices for random shuffle.
        self.idx = list(range(len(self.captions)))
        self.on_epoch_end()

    def on_epoch_end(self):
        logging.info("Data shuffle.")
        random.shuffle(self.idx)

    def __len__(self):
        return len(self.captions) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.idx[self.batch_size * idx:(idx + 1) * self.batch_size]
        partial_captions = []
        next_words = []
        images = []

        for idx in batch_idx:
            crt_img = self.encodings[self.image_ids[idx]]
            caption = self.captions[idx]

            partial_captions.append(
                [self.word_index[w] for w in caption[:-1] if w in self.word_index])
            pred = np.zeros((self.max_length, self.max_vocab_index))
            for i in range(len(caption) - 1):
                word = caption[i + 1]
                if word in self.word_index:
                    pred[i, self.word_index[caption[i + 1]]] = 1

            next_words.append(pred)
            images.append(crt_img)

        next_words = np.asarray(next_words)
        partial_captions = sequence.pad_sequences(
            partial_captions,
            maxlen=self.max_length, padding='post')
        images = np.asarray(images)
        return [[images, partial_captions], next_words]

    @staticmethod
    def _load_captions(captions_path):
        captions = []
        image_ids = []
        with open(captions_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                image_ids.append(row[0])
                captions.append(row[1].split())
        return captions, image_ids

    @staticmethod
    def _generate_index_word(captions):
        tokens = []
        for caption in captions:
            tokens.extend(caption)
        tokens = list(set(tokens))
        return {(i + 1): w for i, w in enumerate(tokens)}

    @staticmethod
    def _generate_word_index(index_word):
        w2i = {}
        for i, w in index_word.items():
            w2i[w] = i
        return w2i
