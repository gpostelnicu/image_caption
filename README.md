# Image Caption

Image Captioning training and inference.


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
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-images --image-ids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.testImages.txt --im-dir data/flickr8k/dataset/Flickr8k_Dataset --output-encodings data/gen1/test_image_encodings.pkl
```

* To generate captions:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-text --image-captions-path data/flickr8k/dataset/Flickr8k_text/Flickr8k.token.txt --imids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.trainImages.txt --output-path data/gen1/train_captions.tsv
```

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py encode-text --image-captions-path data/flickr8k/dataset/Flickr8k_text/Flickr8k.token.txt --imids-path data/flickr8k/dataset/Flickr8k_text/Flickr_8k.testImages.txt --output-path data/gen1/test_captions.tsv
```

* To run training:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python bin/train.py train --training-captions-path data/gen1/train_captions.tsv --test-captions-path data/gen1/test_captions.tsv --train-image-encodings-path data/gen1/train_image_encodings.pkl --test-image-encodings-path data/gen1/test_image_encodings.pkl --num-epochs 50 --output-path data/gen1/model1.h5
```

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

* _Cookiecutter: https://github.com/audreyr/cookiecutter
* _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
