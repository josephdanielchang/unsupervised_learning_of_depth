# Changed rb to r to fix a bug
# Added code to output scaled raw data
# Preliminary attempt to use GPU

from __future__ import division
import tensorflow as tf
import numpy as np
import os
#from matplotlib import pyplot as plt #
# import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")                 #default: 128
flags.DEFINE_integer("img_width", 416, "Image width")                   #default: 416
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

def main(_):
    with open('data/cam_1_1/cam_1_1_trim_zoom_2.txt', 'r') as f:               #r --> rb if read as byte without auto decoding
        test_files = f.readlines()
#        test_files = [t.decode() for t in test_files]                       #decodes bytes to strings
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS.output_dir):                                #make output dir if none exists
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)                #basename = checkpoint model name
    output_file = FLAGS.output_dir + '/' + basename             #name output npy file
    output_scale_images = FLAGS.output_dir + '/scaled_raw_data' ###
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto(log_device_placement=True) #
    config.gpu_options.allow_growth = True
    #with tf.device('/device:GPU:0'):
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        scaled_raw_all = []                                                         ###
        for t in range(0, len(test_files), FLAGS.batch_size):
            print(test_files[t])                                                    #print every 4 frame number
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))      #print every 100 frames
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),           #inputs = batchsize,height,width parameters
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):                                          #exit if frame # + batch size greater than total frames
                    break
                fh = open(test_files[idx], 'rb')                                    #reads text as bytes
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)   #scale raw image to width,height parameters
                inputs[b] = np.array(scaled_im)                                                 #input = array of scaled raw image

                ### ADDED TO SAVE SCALED RAW DATA
                scaled_raw_all.append(inputs[b])

                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            pred = sfm.inference(inputs, sess, mode='depth')                                   #predict depth
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):                                                     #exit if frame # + batch size greater than total frames   
                    break
                pred_all.append(pred['depth'][b,:,:,0])
        np.save(output_scale_images, scaled_raw_all)                                            ###
        np.save(output_file, pred_all)

if __name__ == '__main__':
    tf.app.run()
