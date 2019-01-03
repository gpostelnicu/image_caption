from collections import namedtuple

from keras.applications import *


CnnArchitecture = namedtuple('CnnArchitecture', ['model', 'preprocess_fn'])

CNN_ARCHITECTURES = {
    'vgg16': CnnArchitecture(model=VGG16, preprocess_fn=vgg16.preprocess_input),
    'resnet50': CnnArchitecture(model=ResNet50, preprocess_fn=resnet50.preprocess_input)
}
