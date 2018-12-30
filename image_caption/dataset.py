import csv
import logging
import pickle
import random

import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import Sequence, to_categorical


class Flickr8kDataset(object):
    def __init__(self, captions_path):
        self.captions, self.image_ids = self._load_captions(captions_path)
        self.max_length = max(len(cap) for cap in self.captions)

    def __len__(self):
        return len(self.captions)

    @staticmethod
    def _load_captions(captions_path):
        captions = []
        image_ids = []
        with open(captions_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                image_ids.append(row[0])
                captions.append(text_to_word_sequence(row[1]))
        return captions, image_ids


class Flickr8kEncodedSequence(Sequence):
    def __init__(self, flickr_dataset, batch_size, encodings_path, tokenizer, max_length=None, num_image_versions=5):
        """
        output_type can be 'word' or 'sequence'
        """
        self.batch_size = batch_size
        self.ds = flickr_dataset
        self.encodings = pickle.load(open(encodings_path, 'rb'))
        self.tok = tokenizer
        self.max_vocab_size = 1 + len(self.tok.index_word)
        self.max_length = max_length
        self.num_image_versions = num_image_versions

        # Indices for random shuffle.
        self.idx = list(range(len(self.ds)))

    def __len__(self):
        return len(self.ds)

    def on_epoch_end(self):
        random.shuffle(self.idx)

    def __getitem__(self, item):
        batch_idx = self.idx[self.batch_size * item:(item + 1) * self.batch_size]

        partial_captions = []
        outputs = []
        images = []

        for idx in batch_idx:
            crt_img = self._get_encodings_image(self.ds.image_ids[idx])
            seq_caption = self.tok.texts_to_sequences(self.ds.captions[idx])

            partial_captions.append(seq_caption[:-1])
            pred = np.zeros(((self.max_length, self.max_vocab_size)))
            for i in range(len(seq_caption) - 1):
                pred[i, seq_caption[i + 1]] = 1

            outputs.append(pred)
            images.append(crt_img)

        outputs = np.asarray(outputs)
        partial_captions = sequence.pad_sequences(
            partial_captions,
            maxlen=self.max_length, padding='post')
        images = np.asarray(images)
        return [[images, partial_captions], outputs]

    def _get_encodings_image(self, imid):
        i = random.randint(0, self.num_image_versions)
        full_id = '{}-{}'.format(imid, i)
        encoding = self.encodings[full_id]
        return encoding



class Flickr8kNextWordSequence(Sequence):
    def __init__(self, flickr_dataset, batch_size, encodings_path, tokenizer, max_length):
        """
        output_type can be 'word' or 'sequence'
        """
        self.batch_size = batch_size
        self.ds = flickr_dataset
        self.encodings = pickle.load(open(encodings_path, 'rb'))
        self.tok = tokenizer
        self.max_vocab_size = 1 + len(self.tok.index_word)
        self.max_length = max_length

        self.ds_prev, self.ds_imid, self.ds_next = self._split_captions()

        # Indices for random shuffle.
        self.idx = list(range(len(self.ds_prev)))

    def __len__(self):
        return len(self.idx)

    def on_epoch_end(self):
        random.shuffle(self.idx)

    def __getitem__(self, item):
        batch_idx = self.idx[self.batch_size * item:(item + 1) * self.batch_size]

        partial_captions = [self.ds_prev[i] for i in batch_idx]
        images = [self.encodings[self.ds_imid[i]] for i in batch_idx]
        next_word = []

        for idx in batch_idx:
            idx_next = self.ds_next[idx]
            pred = np.zeros(self.max_vocab_size)
            pred[idx_next] = 1.
            next_word.append(pred)

        partial_captions = sequence.pad_sequences(partial_captions, maxlen=self.max_length, padding='post')
        out = to_categorical(next_word, num_classes=self.max_vocab_size)
        images = np.asarray(images)
        return [[images, partial_captions], out]

    def _split_captions(self):
        out_prev = []
        out_imid = []
        out_next = []

        seq_captions = self.tok.texts_to_sequences(self.ds.captions)
        for i in range(len(self.ds)):
            imid = self.ds.image_ids[i]
            caption = seq_captions[i]

            for j in range(1, len(caption)):
                out_prev.append(caption[:j])
                out_imid.append(imid)
                out_next.append(caption[j])
        return out_prev, out_imid, out_next


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
        if self.index_word is None:
            raise ValueError("Cannot retrieve items without setting index_word.")
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
                    pred[i, self.word_index[word]] = 1

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
