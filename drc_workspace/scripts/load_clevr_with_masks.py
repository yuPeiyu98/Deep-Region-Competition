# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CLEVR (with masks) dataset reader."""

import tensorflow as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.io.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.io.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.io.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.io.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.

  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.

  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

if __name__ == '__main__':  
  import cv2
  import numpy as np

  tf.compat.v1.disable_eager_execution()

  tf_records_path = './clevr_with_masks_train.tfrecords'
  batch_size = 1

  dataset = dataset(tf_records_path)
  batched_dataset = dataset.batch(batch_size)  # optional batching
  iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)
  data = iterator.get_next()

  with tf.compat.v1.train.SingularMonitoredSession() as sess:
    cnt = 0
    while True:
      try:
        d = sess.run(data)
        print(cnt)

        np.savez('./meta/{}.npz'.format(cnt), **d)
        cnt += 1
      except tf.errors.OutOfRangeError:
        break   
    print(cnt) 
