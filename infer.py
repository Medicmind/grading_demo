# Medicmind
# Sample execution using default pneumonia frozen model is: 
# python infer.py --checkpoint_dir="camera/data/stripped_model.pb" --filename="12B_1.jpg"


"""A library to evaluate frozen Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time

from PIL import Image
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf


tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('filename', '',
                           """File to test.""")

FLAGS = tf.app.flags.FLAGS

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def evaluate(filename):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():

    print("-----------Filename: "+filename)

    images = []
    image = Image.open(filename)

    width=image.size[0]
    height=image.size[1]

    # This is to mimic the center crop of image_processingeuc.py 
    # Why do we use this
   
    flt=1 #0.875
    w=int(width*flt)
    h=int(height*flt)
    x=int((width-w)/2)
    y=int((height-h)/2)
    image.crop((x,y,w,h))

##   Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image=image.resize((299,299),Image.ANTIALIAS)
    #image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC) #INTER_LINEAR)
    images.append(image)
    images = np.array(image, dtype=np.uint8)
    images = images.astype('float32')

    images = np.multiply(images, 1.0/255.0)

  
    images = np.subtract(images, 0.5)
    images = np.multiply(images, 2.0)
##   The input to the network is of shape [None image_size image_size num_channels]. 
## Hence we reshape.
 
    x_batch = images.reshape(1, 299,299,3) #num_channels)
 


    graph = load_graph(FLAGS.checkpoint_dir)
    logits=graph.get_tensor_by_name('prefix/inception_v3/logits/logits/xw_plus_b:0')
    x= graph.get_tensor_by_name('prefix/Reshape:0')
    with tf.Session(graph=graph) as sess:


      feed_dict = {x : x_batch}
      prediction= sess.run(logits,feed_dict=feed_dict)
    
      print('Score :'+str(max(0,min(1,prediction[0][0])))+" for image "+filename)
 


def main(unused_argv=None):
  evaluate(FLAGS.filename)


if __name__ == '__main__':
  tf.app.run()

