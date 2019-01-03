# Image Caption

Image Captioning training and inference.

Example networks generating image captions. Trained on the [Flickr8K](https://forms.illinois.edu/sec/1713398) dataset.

Example results:

| Image | Caption |
|-------|---------|
| ![alt text](https://github.com/gpostelnicu/image_caption/images/images/890734502_a5ae67beac.jpg) | a young boy in a swimsuit is splashing in the water |
| ![alt text](https://github.com/gpostelnicu/image_caption/images/images/1304100320_c8990a1539.jpg) | a dog runs through the grass |


* Free software: MIT license
* Documentation: https://image-caption.readthedocs.io.


## Features

* TODO

## Commands:

* To generate image encodings:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-images --image-ids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.trainImages.txt --im-dir data/flickr8k/dataset/Flickr8k_Dataset --output-encodings data/gen2/train_image_encodings.pkl
```

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-images --image-ids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.testImages.txt --im-dir data/flickr8k/dataset/Flickr8k_Dataset --output-encodings data/gen2/test_image_encodings.pkl
```

* To generate captions:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-text --image-captions-path data/flickr8k/dataset/Flickr8k_text/Flickr8k.token.txt --imids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.trainImages.txt --output-path data/gen2/train_captions.tsv
```

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-text --image-captions-path data/flickr8k/dataset/Flickr8k_text/Flickr8k.token.txt --imids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.testImages.txt --output-path data/gen2/test_captions.tsv
```

* To run training:

** for full LSTM prediction:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py train --training-captions-path data/gen2/train_captions.tsv --test-captions-path data/gen2/test_captions.tsv --train-image-encodings-path data/gen2/train_image_encodings.pkl --test-image-encodings-path data/gen2/test_image_encodings.pkl --num-epochs 100 --output-prefix data/gen3/lstm_big --batch-size 128 --learning-rate 1e-4 --lstm-units 1024 --embedding-dim 300 --dropout .5 --recurrent-dropout .5 --embeddings-path data/fasttext/crawl-300d-2M.vec
```

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py train_out_w2v --training-captions-path data/gen2/train_captions.tsv --test-captions-path data/gen2/test_captions.tsv --train-image-encodings-path data/gen2/train_image_encodings.pkl --test-image-encodings-path data/gen2/test_image_encodings.pkl --num-epochs 100 --output-prefix data/gen2/out19 --batch-size 1024 --learning-rate 1e-5 --lstm-units 128 --embedding-dim 300 --dropout .5 --recurrent-dropout .5 --num-dense-layers 2 --embeddings-path data/fasttext/crawl-300d-2M.vec
```

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py train_out_onehot --training-captions-path data/gen2/train_captions.tsv --test-captions-path data/gen2/test_captions.tsv --train-image-encodings-path data/gen2/train_image_encodings.pkl --test-image-encodings-path data/gen2/test_image_encodings.pkl --num-epochs 100 --output-prefix data/gen3/oh_merge_2 --batch-size 1024 --learning-rate 1e-4 --lstm-units 256 --embedding-dim 300 --dropout.5 --recurrent-dropout .5 --num-dense-layers 2 --image-dense-dim 256 --embeddings-path data/fasttext/crawl-300d-2M.vec --num-lstm-layers 1
```

* To perform inference:

```bash
BASE=oh_merge; PYTHONPATH=`pwd`:$PYTHONPATH  python bin/train.py inference2 --im-path data/flickr8k/dataset/Flickr8k_Dataset/2610447973_89227ff978.jpg  --model-path data/gen3/${BASE}_model.h5 --tok-path data/gen3/${BASE}-tok.pkl
```

which yields: "a woman in a blue shirt is walking down a sidewalk".

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* _Cookiecutter: https://github.com/audreyr/cookiecutter
* _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
