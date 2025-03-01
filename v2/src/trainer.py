import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf


def callback(log_dir, ckpt_dir):
  csv_logger = tf.keras.callbacks.CSVLogger(
      log_dir + 'log.csv',
      separator=",",
      append=True
  )

  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
      ckpt_dir+'checkpoint.model.keras',
      monitor="val_loss",
      verbose=0,
      mode="min",
  )

  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss",
      patience=5,
      verbose=1,
      mode="min",
      restore_best_weights=True,
      start_from_epoch=10,
  )

  callbacks = [
      csv_logger,
      model_checkpoint,
      early_stopping,
  ]

  return callbacks


def trainer(model,
            train_dataset,
            validation_dataset,
            num_epochs,
            ckpt_dir,
            log_dir,
            continue_training):

  if continue_training == False:
    initial_epoch = 0
    try:
      shutil.rmtree(log_dir)
    except:
      print('directory not found')
    os.mkdir(log_dir)
  else:
    model.load_weights(ckpt_dir+'checkpoint.model.keras')
    df = pd.read_csv(log_dir+'log.csv')
    initial_epoch = df['epoch'].values[-1]

  unbatch_train_ds = train_dataset.unbatch()
  true_masks = list(unbatch_train_ds.map(lambda x, y: y))

  pixel_0 = pixel_1 = pixel_2 = 0
  for mask in true_masks:
    pixel_0 += np.sum(mask == 0)
    pixel_1 += np.sum(mask == 1)
    pixel_2 += np.sum(mask == 2)

  total_pixel = np.sum([pixel_0, pixel_1, pixel_2])

  weight_for_0 = pixel_0 / total_pixel
  weight_for_1 = pixel_1 / total_pixel
  weight_for_2 = pixel_2 / total_pixel

  print('Weight for class 0: {:.10f}'.format(weight_for_0))
  print('Weight for class 1: {:.10f}'.format(weight_for_1))
  print('Weight for class 2: {:.10f}'.format(weight_for_2))

  def add_sample_weights(image=0, label=0):
    class_weights = tf.constant([weight_for_0, weight_for_1, weight_for_2])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    class_weights = tf.constant([weight_for_0, weight_for_1, weight_for_2])
    class_weights = class_weights/tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return image, label, sample_weights

  model.fit(
      train_dataset.map(add_sample_weights),
      epochs=num_epochs,
      initial_epoch=initial_epoch,
      callbacks=callback(log_dir, ckpt_dir),
      validation_data=validation_dataset,
  )

  model.save(log_dir + 'model.keras')

  return model
