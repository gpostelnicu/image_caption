import numpy as np

from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


class ImageEncoder(object):
    def __init__(self, random_transform=False):
        self.model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

        self.datagen = None
        if random_transform:
            self.datagen = ImageDataGenerator(
                rotation_range=2.,
                zoom_range=.02)

    def process(self, im):
        if isinstance(im, str):
            im = image.load_img(im, target_size=(224, 224))

        x = image.img_to_array(im)
        if self.datagen is not None:
            x = self.datagen.random_transform(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        norm_im = np.asarray(x)
        prediction = self.model.predict(norm_im)
        prediction = np.reshape(prediction, prediction.shape[1])
        return prediction
