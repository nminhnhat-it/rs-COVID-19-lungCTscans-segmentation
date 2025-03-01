import argparse
import os
import yaml

import tensorflow as tf

from src.dataloader import load_data, tf_dataset
from src.trainer import trainer
from src.validator import validator
from src.model import build_model


def load_config(config_file):
  with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
  return config


if __name__ == '__main__':
  # Use argparse to receive parameters from the command line
  parser = argparse.ArgumentParser(description="Train Unet3+ Model")
  parser.add_argument('--config', default='config.yaml', help="Path to config file")
  args = parser.parse_args()

  config = load_config(args.config)

  # Get parameters from config
  batch_size = config['batch_size']
  num_epochs = config['num_epochs']
  learning_rate = config['learning_rate']
  input_shape = config['input_shape']
  augment = config['augment']
  ckpt_dir = config['ckpt_dir']
  log_dir = config['log_dir']
  data_root_path = config['data_root_path']
  encoder = config['encoder']
  fine_tune_at = config['fine_tune_at']
  continue_training = config['continue_training']
  gradient_accumulation_steps = config['gradient_accumulation_steps']
  training = config['training']
  model_summary = config['model_summary']

  # Ensure checkpoint directory exists
  if not os.path.exists(config['ckpt_dir']):
    os.makedirs(config['ckpt_dir'])

  # Ensure logs directory exists
  if not os.path.exists(config['log_dir']):
    os.makedirs(config['log_dir'])

  # Check if GPU is available
  device_name = tf.test.gpu_device_name()
  if device_name:
    print(f'Using GPU: {device_name}')
  else:
    print('Using CPU')

  (train_images, train_masks), (test_images, test_masks), (validation_images, validation_masks) = load_data(data_root_path)

  train_dataset, train_num_samples = tf_dataset(train_images, train_masks, batch_size, input_shape, augment=augment)
  test_dataset, test_num_samples = tf_dataset(test_images, test_masks, batch_size, input_shape, augment=False)
  validation_dataset, val_num_samples = tf_dataset(validation_images, validation_masks, batch_size, input_shape, augment=False)

  model = build_model(input_shape, 3, encoder, fine_tune_at, learning_rate, gradient_accumulation_steps)
  if model_summary:
    model.summary()

  if training:
    model = trainer(model, train_dataset, validation_dataset, num_epochs, ckpt_dir, log_dir, continue_training)

  validator(model, test_dataset, test_num_samples, log_dir, batch_size)
