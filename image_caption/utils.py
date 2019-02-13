import logging
import string
import sys

import numpy as np


def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_fasttext(embeddings_path):
    embeddings_index = {}
    with open(embeddings_path) as f:
        num_words, emb_size = [int(i) for i in f.readline().strip().split()]
        print("Found {} words in embedding of size {}".format(num_words, emb_size))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def create_embedding_matrix(index, embeddings, dim, special_tokens, init_random=False, start_at_one=False):
    vec_dim = dim + len(special_tokens)

    # Indices from tokenizer are 1-based.
    max_word_idx = max(index.values())
    idx_offset = -1
    if start_at_one:
        max_word_idx += 1
        idx_offset = 0
    logging.info("Setting max word idx at: {}".format(max_word_idx))

    if init_random:
        print('Embedding random initializer.')
        matrix = np.random.random((max(index.values()) + 1, vec_dim))
    else:
        print('Embedding zeros initializer.')
        matrix = np.zeros((max(index.values()) + 1, vec_dim))
    num_words_in_embedding = 0
    num_lowercase = 0
    num_random = 0

    matrix[0] = np.zeros((vec_dim,))
    padding = np.zeros((len(special_tokens,)))
    for word, i in index.items():
        i += idx_offset
        if i < 0:  # remove punctuation.
            continue
        vec = embeddings.get(word)
        if vec is not None:
            num_words_in_embedding += 1
            matrix[i] = np.concatenate((vec, padding))
        elif word.lower() in embeddings:
            num_lowercase += 1
            vec = embeddings.get(word.lower())
            matrix[i] = np.concatenate((vec, padding))
        elif word in special_tokens:
            vec = np.zeros((vec_dim,))
            vec[dim + special_tokens.index(word)] = 1.
        else:
            prime1 = 197
            prime2 = 97
            v = np.zeros((vec_dim,))
            v[i % prime1] = 1.
            v[i % prime2 + prime1] = -1.
            matrix[i] = v
            num_random += 1
    print('Num words found:', num_words_in_embedding, ' lower case: ', num_lowercase, ' num random: ', num_random)
    # Normalize
    matrix = matrix / np.max(np.abs(matrix))

    return matrix


def normalize_embeddings(emb):
    for word in emb:
        v = emb[word]
        emb[word] = v / np.linalg.norm(v)


def write_embeddings(dic, fname, embedding_size=300):
    with open(fname, 'w') as fh:
        fh.write('{} {}\n'.format(len(dic), embedding_size))
        for k in dic:
            v = dic[k]
            fh.write('{} {}\n'.format(k, ' '.join(str(f) for f in v)))
        fh.close()


def cleanup_caption(caption):
    table = str.maketrans(string.punctuation, ''.join([' ' for _ in range(len(string.punctuation))]))
    words = caption.translate(table)
    words = words.split(' ')
    words = [w for w in words if len(w) > 1]
    words = [w for w in words if w.isalpha()]
    return ' '.join(words)


def filter_tokenizer(tok, min_docs_per_word=5):
    words_to_keep = [k for k, v in tok.word_docs.items() if v >= min_docs_per_word]
    tok.word_index = dict([(w, i + 1) for i, w in enumerate(words_to_keep)])
    tok.index_word = {v:k for k, v in tok.word_index.items()}


