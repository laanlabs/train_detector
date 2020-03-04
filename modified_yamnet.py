
"""
Slightly modified yamnet keras Model that outputs 
the dense feature vectors along with the class predictions

Originally: 
TF_MODELS_REPO/models/research/audioset/yamnet/yamnet.py

"""

import sys

yamnet_base = './models/research/audioset/yamnet/'
sys.path.append(yamnet_base)

import os
assert os.path.exists(yamnet_base)

# yamnet imports 
import params
#import modified_yamnet as yamnet_model
import features as features_lib

# TF / keras 
from tensorflow.keras import Model, layers
import tensorflow as tf

from yamnet import _YAMNET_LAYER_DEFS



def yamnet(features):
  """Define the core YAMNet mode in Keras."""
  net = layers.Reshape(
    (params.PATCH_FRAMES, params.PATCH_BANDS, 1),
    input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(net)
  net = layers.GlobalAveragePooling2D()(net)
  logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
  predictions = layers.Activation(
    name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
    activation=params.CLASSIFIER_ACTIVATION)(logits)
  return predictions, net


def yamnet_frames_model(feature_params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    feature_params: An object with parameter fields to control the feature
    calculation.

  Returns:
    A model accepting (1, num_samples) waveform input and emitting a
    (num_patches, num_classes) matrix of class scores per time frame as
    well as a (num_spectrogram_frames, num_mel_bins) spectrogram feature
    matrix.
  """
  waveform = layers.Input(batch_shape=(1, None))
  # Store the intermediate spectrogram features to use in visualization.
  spectrogram = features_lib.waveform_to_log_mel_spectrogram(
    tf.squeeze(waveform, axis=0), feature_params)
  patches = features_lib.spectrogram_to_patches(spectrogram, feature_params)
  predictions, net = yamnet(patches)
  frames_model = Model(name='yamnet_frames', 
                       inputs=waveform, outputs=[predictions, spectrogram, net, patches])
  return frames_model, net