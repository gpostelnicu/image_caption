import pickle

from keras.models import load_model
from keras.preprocessing import image, sequence
import numpy as np


class WordInference(object):
    def __init__(self, model_path, tok_path, max_cap_len=39, target_size=(224, 224)):
        with open(tok_path, 'rb') as fh:
            self.tok = pickle.load(fh)
        self.model = load_model(model_path)
        self.max_cap_len = max_cap_len
        self.target_size = target_size

    def _encode_partial_caption(self, partial_caption, im_arr):
        input_text = [[self.tok.word_index[w] for w in partial_caption if w in self.tok.word_index]]
        input_text = sequence.pad_sequences(input_text, maxlen=self.max_cap_len, padding='post')
        return [np.array([im_arr]), input_text]

    def process_image(self, im_path=None, im_arr=None):
        if im_path is not None:
            im = image.load_img(im_path, target_size=self.target_size)
            im_arr = image.img_to_array(im)
        else:
            assert im_arr is not None

        partial_cap = []
        EOS_TOKEN = 'endtoken'

        while True:
            inputs = self._encode_partial_caption(partial_cap, im_arr)
            preds = self.model.predict(inputs)
            pred_idx = np.argmax(preds, axis=-1)[0][0]
            pred_word = self.tok.index_word[pred_idx + 1]  # Shift by 1 (works for Flickr8kImageSequence)
            if pred_word == EOS_TOKEN or len(partial_cap) == self.max_cap_len:
                break
            partial_cap.append(pred_word)
        return partial_cap

