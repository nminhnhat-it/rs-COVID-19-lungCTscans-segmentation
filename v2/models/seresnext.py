import tensorflow as tf
from tensorflow import keras as k
from classification_models.tfkeras import Classifiers


def seresnext(input_shape,
           fine_tune_at=False):
  """ Resnet model """
  filters = [64, 256, 512, 1024, 2048]

  seresnext, preprocess_input = Classifiers.get('seresnext50')
  seresnext_model = seresnext(include_top=False, input_shape=input_shape, weights='imagenet')

  seresnext_model.trainable = True
  print("Number of layers in the base model: ", len(seresnext_model.layers))
  fine_tune_at = round(len(seresnext_model.layers) * fine_tune_at)
  if fine_tune_at != False:
    for layer in seresnext_model.layers[:fine_tune_at]:
      layer.trainable = False

  return seresnext_model, filters
