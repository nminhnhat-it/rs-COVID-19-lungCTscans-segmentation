import tensorflow as tf
import keras_cv
from glob import glob
import os
from functools import partial

from src.augment import Augment


def load_data(path):
  train_images = sorted(glob(os.path.join(path, 'train', 'images', '*')))
  train_masks = sorted(glob(os.path.join(path, 'train', 'masks', '*')))

  test_images = sorted(glob(os.path.join(path, 'test', 'images', '*')))
  test_masks = sorted(glob(os.path.join(path, 'test', 'masks', '*')))

  validation_images = sorted(glob(os.path.join(path, 'val', 'images', '*')))
  validation_masks = sorted(glob(os.path.join(path, 'val', 'masks', '*')))

  return (train_images, train_masks), (test_images, test_masks), (validation_images, validation_masks)


def read_image(path, input_shape):
  path = path.decode()

  img = tf.keras.utils.load_img(
      path,
      color_mode="rgb",
      target_size=(input_shape[0], input_shape[1]),
      interpolation="bilinear",
  )
  img = tf.keras.utils.img_to_array(img, dtype='float32')
  return img


def read_mask(path, input_shape):
  path = path.decode()

  img = tf.keras.utils.load_img(
      path,
      color_mode="grayscale",
      target_size=(input_shape[0], input_shape[1]),
      interpolation="bilinear",
  )
  img = tf.keras.utils.img_to_array(img, dtype='float32')
  return img


def tf_parse(image, mask, input_shape):
  def _parse(image, mask):
    image = read_image(image, input_shape)
    mask = read_mask(mask, input_shape)
    return image, mask

  image, mask = tf.numpy_function(_parse, [image, mask], [tf.float32, tf.float32])
  image.set_shape([input_shape[0], input_shape[1], input_shape[2]])
  image = tf.cast(image, tf.float32) / 255.0
  mask.set_shape([input_shape[0], input_shape[1], 1])
  return image, mask


def tf_dataset(image, mask, batch_size, input_shape, augment=False):
  dataset = tf.data.Dataset.from_tensor_slices((image, mask))
  dataset = dataset.map(partial(tf_parse, input_shape=input_shape), num_parallel_calls=tf.data.AUTOTUNE)
  if (augment != False):
    dataset_augmented = dataset.map(Augment(augment['random_flip'], augment['random_cutout'], augment['random_contrast'], augment['random_brightness']), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.concatenate(dataset_augmented)
  num_samples = len(dataset)
  dataset = dataset.shuffle(num_samples)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset, num_samples
