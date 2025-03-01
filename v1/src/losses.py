"""
Implementation of different loss functions
"""
import tensorflow as tf
import tensorflow.keras.backend as K


num_classes = 2

def init_num_classes(init_num_classes):
  global num_classes
  num_classes = init_num_classes


def iou(y_true, y_pred, smooth=K.epsilon()):
  y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[..., 1:])
  y_pred_f = K.flatten(y_pred[..., 1:])
  intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
  union = K.sum(y_true_f + y_pred_f, axis=-1)
  union = union - intersection
  iou = K.mean((intersection + smooth) / (union + smooth))
  return iou


def dice_coef(y_true, y_pred, smooth=K.epsilon()):
  y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=num_classes)[..., 1:])
  y_pred_f = K.flatten(y_pred[..., 1:])
  intersect = K.sum(y_true_f * y_pred_f, axis=-1)
  denom = K.sum(y_true_f + y_pred_f, axis=-1)
  return K.mean((2. * intersect / (denom + smooth)))


class dice_coef_loss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()

  def call(self, y_true, y_pred):
    dice = 1 - dice_coef(y_true, y_pred)
    return dice
