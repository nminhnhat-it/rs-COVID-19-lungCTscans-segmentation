import tensorflow as tf
import keras_cv


class Augment(tf.keras.layers.Layer):
  def __init__(self,
               random_flip,
               random_cutout,
               random_contrast,
               random_brightness):
    super().__init__()
    self.random_flip = random_flip
    self.random_cutout = random_cutout
    self.random_contrast = random_contrast
    self.random_brightness = random_brightness

    if random_flip:
      self.augment_RandomFlip = tf.keras.layers.RandomFlip(mode='horizontal')
    if random_cutout:
      self.augment_RandomCutout = keras_cv.layers.RandomCutout(height_factor=0.1, width_factor=0.1, fill_mode="constant", fill_value=0.0)
    if random_contrast:
      self.augment_RandomContrast = tf.keras.layers.RandomContrast(factor=0.2)
    if random_brightness:
      self.augment_RandomBrightness = tf.keras.layers.RandomBrightness(factor=0.2, value_range=[0.0, 1.0])

  def call(self, inputs, labels):
    labels = labels[:, :, 0]
    labels = tf.stack([labels, labels, labels], -1)

    labels = tf.cast(labels, 'float32')
    output = tf.concat([inputs, labels], -1)

    if self.random_flip:
      output = self.augment_RandomFlip(output)
    if self.random_cutout:
      for i in range(20):
        output = self.augment_RandomCutout(output)

    labels_augmented = output[:, :, 4]
    labels_augmented = tf.expand_dims(tf.cast(labels_augmented, 'float32'), axis=-1)

    if self.random_contrast:
      output = self.augment_RandomContrast(output)
    if self.random_brightness:
      output = self.augment_RandomBrightness(output)

    inputs_augmented = output[:, :, 0:3]

    return inputs_augmented, labels_augmented