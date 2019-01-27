import os

import numpy as np
import pickle

import fire
from keras.models import load_model
from keras.preprocessing import image
import nltk
from keras.preprocessing.sequence import pad_sequences

from image_caption.dataset import Flickr8kDataset


def evaluate(model_path, tokenizer_path, captions_path, images_dir,
             max_cap_len=39, include_start_token=True, verbose=False, top_n=-1):
    tok = pickle.load(open(tokenizer_path, 'rb'))
    model = load_model(model_path)
    flkr = Flickr8kDataset(captions_path=captions_path)

    def encode_partial_cap(partial_cap, im):
        input_text = [[tok.word_index[w] for w in partial_cap if w in tok.word_index]]
        input_text = pad_sequences(input_text, maxlen=max_cap_len, padding='post')
        im = np.array([im])
        return [im, input_text]

    EOS_TOKEN = 'endtoken'
    scores_bleu1 = []
    scores_bleu2 = []
    for imid, caption in flkr:
        img = image.load_img(os.path.join(images_dir, imid), target_size=(224, 224, 3))
        im_arr = image.img_to_array(img)

        partial_cap = []
        if include_start_token:
            partial_cap.append('starttoken')

        while True:
            inputs = encode_partial_cap(partial_cap, im_arr)
            preds = model.predict(inputs)[0, len(partial_cap), :]
            next_idx = np.argmax(preds, axis=-1)
            next_word = tok.index_word[next_idx]
            if next_word == EOS_TOKEN or len(partial_cap) == 38:
                break
            partial_cap.append(next_word)

        bleu1 = nltk.translate.bleu_score.sentence_bleu([caption[1:]], partial_cap, weights=[1., 0., 0., 0.])
        bleu2 = nltk.translate.bleu_score.sentence_bleu([caption[1:]], partial_cap, weights=[0., 1., 0., 0.])
        scores_bleu1.append(bleu1)
        scores_bleu2.append(bleu2)
        if verbose:
            print('items: {}, Target: {}, predicted: {}, score: {}'.format(
                len(scores_bleu1), caption[1:], partial_cap[:-1], score))
        if top_n > 0 and len(scores_bleu1) >= top_n:
            break

    print('Average score: bleu1 = {}, bleu2 = {}'.format(np.mean(scores_bleu1), np.mean(scores_bleu2)))


if __name__ == '__main__':
    fire.Fire()
