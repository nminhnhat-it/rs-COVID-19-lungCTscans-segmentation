import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import os

from utils.utils import compute_metrics, create_mask, display_images, plot_history


def get_test_image_and_annotation_arrays(test_dataset, test_num_samples):
  ds = test_dataset.unbatch()
  ds_num_img = test_num_samples
  ds = ds.batch(ds_num_img)

  for y_true_images, y_true_segments in ds.take(ds_num_img):
    y_true_images = y_true_images
    y_true_segments = y_true_segments

  return y_true_images, y_true_segments


def show_predictions(model, dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display_images([image[0], mask[0], create_mask(pred_mask)[0]])
      compute_metrics(mask[0], create_mask(pred_mask)[0])


def validator(model, test_dataset, test_num_samples, log_dir, batch_size):
  y_true_images, y_true_segments = get_test_image_and_annotation_arrays(test_dataset, test_num_samples)

  model.load_weights(log_dir + 'model.keras')
  # y_pred_masks = model.predict(y_true_images, batch_size)
  # y_pred_masks = create_mask(y_pred_masks)

  # compute_metrics(y_true_segments, y_pred_masks, True)
  if os.path.exists(log_dir + 'log.csv'):
    plot_history(log_dir + 'log.csv')

  show_predictions(model, test_dataset, num=73)
