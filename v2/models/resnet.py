import tensorflow as tf
import tensorflow.keras as k
from classification_models.tfkeras import Classifiers


def resnet(input_shape,
           fine_tune_at=False):
  """ Resnet model """
  filters = [64, 256, 512, 1024, 2048]

  resnet, _ = Classifiers.get('resnet50')
  resnet_model = resnet(include_top=False, input_shape=input_shape, weights='imagenet')

  resnet_model.trainable = True
  print("Number of layers in the Resnet model: ", len(resnet_model.layers))

  fine_tune_at = round(len(resnet_model.layers) * fine_tune_at)
  if fine_tune_at != False:
    for layer in resnet_model.layers[:fine_tune_at]:
      layer.trainable = False

  return resnet_model, filters
