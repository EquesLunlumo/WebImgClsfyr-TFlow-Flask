from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys

# pylint: disable=unused-import,g-bad-import-order
from six.moves import urllib
import numpy as np
import tensorflow as tf
# pylint: enable=unused-import,g-bad-import-order

from tensorflow.python.platform import gfile

FLAGS = None

def create_graph():
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join('*/tf_files/retrained_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read()) 
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    label_lines = [line.rstrip() for line in tf.gfile.GFile("*/tf_files/retrained_labels.txt")]
    top_k_human = []
    for node_id in top_k:
      human_string = label_lines[node_id]
      score = predictions[0][node_id]
      top_k_human.append({'score': score,
                          'human_string': human_string})
    return top_k_human

# configuration
DEBUG = True
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

static_path = '*/static'
from flask import Flask, request, render_template, url_for
import time
from werkzeug import secure_filename
app = Flask(__name__,
        
            static_folder=static_path,
            static_url_path=static_path)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = static_path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def resize_image(filepath, max_width, max_height):
    import os
    from PIL import Image
    filename, file_extension = os.path.splitext(filepath)
    outfile = "%s.thumbnail.%s" % (filename, file_extension)
    if filepath != outfile:
        try:
            size = (max_width, max_height)
            im = Image.open(filepath)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "JPEG")
            return outfile
        except IOError:
            print("cannot create thumbnail for '%s'" % filepath)
            return None



@app.route('/', methods=['POST', 'GET'])
def home():
    """home classifier."""
    global n
    # if request.method == 'GET' and request.args.get('heartbeat', '') != "":
    #     return request.args.get('heartbeat', '')

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(new_path)
            new_path = resize_image(new_path, 500, 500)
            if new_path is not None:
                print("#"*80)
                ret = run_inference_on_image(new_path)
                for el in ret:
                    print("%s: %0.4f" % (el['human_string'], el['score']))
                thumbname = os.path.basename(new_path)
                return render_template('predict.html',
                                       ret=ret,
                                       img_src=url_for('static',
                                                       filename=thumbname))
    # else:
    #     # Page where the user can enter a recording
    return render_template('home.html')


def main(_):
  if not os.path.exists('static'):
    os.makedirs('static')
  app.run()


if __name__ == '__main__':
  tf.app.run()