
"""
Usage:
  # Create train data:
  python data_conversion.py --csv_input=data/train_labels.csv  --output_path=eval.record
  # Create test data:
  python data_conversion.py --csv_input=data/test_labels.csv  --output_path=test.record
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import sys
import os
import PIL.Image
import io
import random


flags = tf.app.flags
flags.DEFINE_boolean('eval', False, 'Root directory for the dataset.')
flags.DEFINE_string('data_dir', '', 'Root directory for the dataset.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_path', '', 'Path to input data')
flags.DEFINE_string('label_map_path', 'google-api/data/crowdai_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../models'))

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def generate_tf_examples(_label_map_dict):
  _input_path = FLAGS.input_path
  df = pd.read_csv(_input_path)
  _img_names = list(set(df.Frame.tolist()))
  if FLAGS.eval:
    random.shuffle(_img_names)
    _img_names = _img_names[:5000]
  for img_name in _img_names:
    print("processing image {}".format(img_name))
    boxes = df[df.Frame.isin([img_name])]

    height = 1200
    width = 1920
    full_path = os.path.join(FLAGS.data_dir, img_name)
    with tf.gfile.GFile(full_path, 'rb') as fid:
      encoded_jpg = fid.read()


    xmins = (boxes.xmin / height).tolist()
    xmaxs = (boxes.xmax / height).tolist()
    ymins = (boxes.ymin / width).tolist()
    ymaxs = (boxes.ymax / width).tolist()
    classes_text = boxes.Label.str.encode('utf8').tolist()
    classes = [_label_map_dict[_text] for _text in boxes.Label.tolist()]
    print("xmins {} xmaxs {} ymins {} ymaxs {}".format(boxes.xmin.tolist(), boxes.xmax.tolist(), boxes.ymin.tolist(), boxes.ymax.tolist()))
    print("normalized xmins {} xmaxs {} ymins {} ymaxs {}".format(xmins, xmaxs, ymins, ymaxs))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_name.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    yield tf_example

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  for tf_example in generate_tf_examples(label_map_dict):
    writer.write(tf_example.SerializeToString())
  writer.close()


if __name__ == '__main__':
  tf.app.run()