from collections import namedtuple

from keras.applications import *


CnnArchitecture = namedtuple('CnnArchitecture', ['model', 'preprocess_fn'])

CNN_ARCHITECTURES = {
    'resnet50': CnnArchitecture(model=ResNet50, preprocess_fn=resnet50.preprocess_input),
    'inceptionv3': CnnArchitecture(model=InceptionV3, preprocess_fn=inception_v3.preprocess_input),
    'vgg16': CnnArchitecture(model=VGG16, preprocess_fn=vgg16.preprocess_input)
}
