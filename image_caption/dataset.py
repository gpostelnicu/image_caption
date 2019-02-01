import csv
import logging
import os
import pickle
import random
from functools import lru_cache

import numpy as np

from keras.preprocessing import sequence, image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import Sequence, to_categorical


class Flickr8kDataset(object):
    def __init__(self, captions_path):
        self.captions, self.image_ids = self._load_captions(captions_path)
        self.max_length = max(len(cap) for cap in self.captions)

    def __len__(self):
        return len(self.captions)

    def __iter__(self):
        yield from zip(self.image_ids, self.captions)

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


class Flickr8kImageSequence(Sequence):
    def __init__(self, flickr_dataset, images_dir, batch_size,
                 tokenizer, image_preprocess_fn, max_length=None, random_image_transform=False,
                 replace_words_ratio=0.0, output_weights=False, captions_start_idx=0):
        self.ds = flickr_dataset
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.tok = tokenizer
        self.max_length = max_length
        self.max_vocab_size = 1 + len(self.tok.index_word)
        self.image_preprocess_fn = image_preprocess_fn
        self.captions_start_idx = captions_start_idx

        if random_image_transform:
            self.datagen = ImageDataGenerator(
                rotation_range=2.,
                zoom_range=.02,
                brightness_range=[.8, 1.2],
                width_shift_range=0.1,
                height_shift_range=0.1
            )
        else:
            self.datagen = None

        self.replace_words_ratio = replace_words_ratio
        self.output_weights = output_weights

        # Indices for random shuffle.
        self.idx = list(range(len(self.ds)))

    def __len__(self):
        return len(self.ds) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.idx)

    def __getitem__(self, item):
        batch_idx = self.idx[self.batch_size * item:(item + 1) * self.batch_size]

        image_paths = [os.path.join(self.images_dir, self.ds.image_ids[i]) for i in batch_idx]
        images = [self._read_img(ip) for ip in image_paths]
        if self.datagen is not None:
            images = [self.datagen.random_transform(im) for im in images]
        images = np.stack(images)
        norm_images = self.image_preprocess_fn(images)
        norm_images = np.asarray(norm_images)

        captions = [self.tok.texts_to_sequences(self.ds.captions[idx]) for idx in batch_idx]

        partial_captions = [c[self.captions_start_idx:-1] for c in captions]
        if self.replace_words_ratio > 0:
            def random_replace(lst):
                rand_idx = np.random.choice(len(lst), int(self.replace_words_ratio * len(lst)), replace=False)
                for i in rand_idx:
                    lst[i] = [np.random.randint(1, self.max_vocab_size - 1)]

            for pc in partial_captions:
                random_replace(pc)
        partial_captions = sequence.pad_sequences(partial_captions,
                                                  maxlen=self.max_length - self.captions_start_idx,
                                                  padding='post')
        partial_captions = np.squeeze(partial_captions)

        outputs = []
        if self.output_weights:
            batch_weights = []

        for caption in captions:
            pred = np.zeros((self.max_length, self.max_vocab_size))
            for i, n in enumerate(caption[1:]):
                pred[i, n] = 1.

            outputs.append(pred)

            if self.output_weights:
                output_len = len(caption) - 1
                w = np.concatenate(
                    (np.ones((output_len,)),
                     np.zeros(self.max_length - output_len,)))
                batch_weights.append(w)

        outputs = np.asarray(outputs)
        if self.output_weights:
            batch_weights = np.asarray(batch_weights)
            return [[norm_images, partial_captions], outputs, batch_weights]
        else:
            return [[norm_images, partial_captions], outputs]

    @lru_cache(maxsize=30000)
    def _read_img(self, im_path):
        im = image.load_img(im_path, target_size=(224, 224))
        x = image.img_to_array(im)
        return x



class Flickr8kEncodedSequence(Sequence):
    def __init__(self, flickr_dataset, batch_size, encodings_path,
                 tokenizer, max_length=None, num_image_versions=5):
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
        return len(self.ds) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.idx)

    def __getitem__(self, item):
        batch_idx = self.idx[self.batch_size * item:(item + 1) * self.batch_size]

        images = [self._get_image_encoding(self.ds.image_ids[i]) for i in batch_idx]
        images = np.asarray(images)

        captions = [self.tok.texts_to_sequences(self.ds.captions[idx]) for idx in batch_idx]

        partial_captions = [c[:-1] for c in captions]
        partial_captions = sequence.pad_sequences(partial_captions, maxlen=self.max_length, padding='post')
        partial_captions = np.squeeze(partial_captions)

        outputs = []
        for caption in captions:
            pred = np.zeros((self.max_length, self.max_vocab_size))
            for i, n in enumerate(caption[1:]):
                pred[i, n] = 1.

            outputs.append(pred)
        outputs = np.asarray(outputs)

        return [[images, partial_captions], outputs]

    def _get_image_encoding(self, imid):
        i = random.randint(0, self.num_image_versions - 1)
        full_id = '{}-{}'.format(imid, i)
        encoding = self.encodings[full_id]
        return encoding


class Flickr8kNextWordSequence(Sequence):
    def __init__(self, images_dir, flickr_dataset, batch_size, tokenizer, max_length,
                 image_preprocess_fn, word_embeddings=None, random_image_transform=False):
        """
        output_type can be 'word' or 'sequence'
        """
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.ds = flickr_dataset
        self.tok = tokenizer
        self.max_vocab_size = 1 + len(self.tok.index_word)
        self.max_length = max_length
        self.word_embeddings = word_embeddings
        self.image_preprocess_fn = image_preprocess_fn
        if self.word_embeddings is not None:
            logging.info("Will output word embeddings.")

        self.ds_prev, self.ds_imid, self.ds_next = self._split_captions()

        if random_image_transform:
            self.datagen = ImageDataGenerator(
                rotation_range=2.,
                zoom_range=.02,
                brightness_range=[.8, 1.2],
                width_shift_range=0.1,
                height_shift_range=0.1
            )
        else:
            self.datagen = None

        # Indices for random shuffle.
        self.idx = list(range(len(self.ds_prev)))

    def __len__(self):
        return len(self.idx) // self.batch_size

    def on_epoch_end(self):
        random.shuffle(self.idx)

    def __getitem__(self, item):
        batch_idx = self.idx[self.batch_size * item:(item + 1) * self.batch_size]

        image_paths = [os.path.join(self.images_dir, self.ds.image_ids[i]) for i in batch_idx]
        images = [self._read_img(ip) for ip in image_paths]
        if self.datagen:
            images = [self.datagen.random_transform(im) for im in images]
        images = np.stack(images)
        norm_images = self.image_preprocess_fn(images)
        norm_images = np.asarray(norm_images)

        partial_captions = [self.ds_prev[i] for i in batch_idx]
        partial_captions = sequence.pad_sequences(partial_captions, maxlen=self.max_length, padding='post')

        if self.word_embeddings is not None:
            out = np.array([self._target_fasttext(self.ds_next[i]) for i in batch_idx])
        else:
            out = to_categorical([self.ds_next[i] for i in batch_idx], num_classes=self.max_vocab_size)
        return [[norm_images, partial_captions], out]

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

    def _target_fasttext(self, word_id):
        return self.word_embeddings[word_id]

    @lru_cache(maxsize=30000)
    def _read_img(self, im_path):
        im = image.load_img(im_path, target_size=(224, 224))
        x = image.img_to_array(im)
        return x



class Flickr8KSequence(Sequence):
    def __init__(self, batch_size, encodings_path,
                 captions_path, max_length=None, index_word=None, captions_start_idx=0):
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
        self.captions_start_idx = captions_start_idx

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
                [self.word_index[w] for w in caption[self.captions_start_idx:-1] if w in self.word_index])
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
