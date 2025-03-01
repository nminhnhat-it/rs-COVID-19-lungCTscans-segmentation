import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask


def display_images(display_list):
  plt.figure(figsize=(10, 10))
  title = ['Input Image', 'True Mask',
           'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


def plot_history(csv_path):
  df = pd.read_csv(csv_path)
  df[['epoch', 'loss', 'val_loss']].plot(
      x='epoch',
      y=['loss', 'val_loss'],
      xlabel='epoch',
      ylabel='loss',
      title='Train and Validation Loss Over Epochs'
  )
  plt.legend()
  plt.grid()
  plt.show()


def compute_metrics(y_trues, y_preds, show_ncm=False):
  y_trues = y_trues.numpy()
  y_preds = y_preds.numpy()
  C00 = C01 = C02 = C10 = C11 = C12 = C20 = C21 = C22 = 0

  for y_true, y_pred in zip(y_trues, y_preds):
    C00 += np.sum((y_true == 0) & (y_pred == 0))
    C01 += np.sum((y_true == 0) & (y_pred == 1))
    C02 += np.sum((y_true == 0) & (y_pred == 2))

    C10 += np.sum((y_true == 1) & (y_pred == 0))
    C11 += np.sum((y_true == 1) & (y_pred == 1))
    C12 += np.sum((y_true == 1) & (y_pred == 2))

    C20 += np.sum((y_true == 2) & (y_pred == 0))
    C21 += np.sum((y_true == 2) & (y_pred == 1))
    C22 += np.sum((y_true == 2) & (y_pred == 2))

  TP0 = C00
  TP1 = C11
  TP2 = C22

  FP0 = C01+C02
  FP1 = C10+C12
  FP2 = C20+C21

  FN0 = C10+C20
  FN1 = C01+C21
  FN2 = C02+C12

  precision0 = TP0/(TP0+FP0)
  precision1 = TP1/(TP1+FP1)
  precision2 = TP2/(TP2+FP2)
  precision = [precision0, precision1, precision2]

  recall0 = TP0/(TP0+FN0)
  recall1 = TP1/(TP1+FN1)
  recall2 = TP2/(TP2+FN2)
  recall = [recall0, recall1, recall2]

  dice0 = 2 * TP0 / (2 * TP0 + FN0 + FP0)
  dice1 = 2 * TP1 / (2 * TP1 + FN1 + FP1)
  dice2 = 2 * TP2 / (2 * TP2 + FN2 + FP2)
  dice_score = [dice0, dice1, dice2]

  iou0 = TP0 / (TP0 + FN0 + FP0)
  iou1 = TP1 / (TP1 + FN1 + FP1)
  iou2 = TP2 / (TP2 + FN2 + FP2)
  iou_score = [iou0, iou1, iou2]

  cm = [[C00, C01, C02],
        [C10, C11, C12],
        [C20, C21, C22]]

  display_string_list = ["Mask {}: IOU: {} Dice Score: {}".format(idx, i, dc) for idx, (i, dc) in enumerate(zip(np.round(iou_score, 4), np.round(dice_score, 4)))]
  display_string = "\n\n".join(display_string_list)
  print(display_string)

  display_string_list = ["Mask {}: Precision: {} Recall: {}".format(idx, i, dc) for idx, (i, dc) in enumerate(zip(np.round(precision, 4), np.round(recall, 4)))]
  display_string = "\n\n".join(display_string_list)
  print(f'\n{display_string}')

  print(f"\nMean dice score: {round(np.mean(dice_score),4)}\n")
  print(f"Mean iou: {round(np.mean(iou_score),4)}")

  if show_ncm == True:
    ncm = np.round(cm/np.sum(cm, axis=1).reshape(-1, 1), 4)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sns.heatmap(ncm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 15})
    ax.set_title('Normalized confusion matrix\n\n', fontsize=15)
    ax.set_xlabel('\nPredicted label', fontsize=15)
    ax.set_ylabel('True label ', fontsize=15)
    ax.xaxis.set_ticklabels(['Non lung nor infection', 'Lung', 'Infection'], fontsize=13)
    ax.yaxis.set_ticklabels(['Non lung nor infection', 'Lung', 'Infection'], fontsize=13)
    plt.show()
