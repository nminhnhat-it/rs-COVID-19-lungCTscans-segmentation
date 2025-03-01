import tensorflow as tf

from models.unet3plus import unet3plus
from utils.loss import iou, dice_coef, dice_coef_loss


def build_model(input_shape,
                num_classes,
                encoder,
                fine_tune_at,
                learning_rate,
                gradient_accumulation_steps):
  model = unet3plus(input_shape, num_classes, encoder, fine_tune_at)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate,
          gradient_accumulation_steps=gradient_accumulation_steps
      ),
      loss=dice_coef_loss(),
      metrics=[
          dice_coef,
          iou,
      ],
  )
  return model
