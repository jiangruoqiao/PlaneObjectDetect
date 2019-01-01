#ecoding:utf-8
import tools
import rot_mnist12K_model
import tensorflow as tf
import sys
import numpy as np
import toolstrain
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TRAIN_FILENAME = '/media/ubuntu/2c0e4fe1-8b1d-4526-b308-a7173c23f87e/NWPUTrain.h5'
TEST_FILENAME = '/media/ubuntu/2c0e4fe1-8b1d-4526-b308-a7173c23f87e/NWPUTest.h5'
LOADED_SIZE = 28
DESIRED_SIZE = 32
# model constants
NUMBER_OF_CLASSES = 11
NUMBER_OF_FILTERS = 40
NUMBER_OF_FC_FEATURES = 5120
NUMBER_OF_TRANSFORMATIONS = 8
# optimization constants
BATCH_SIZE = 64
TEST_CHUNK_SIZE = 517
ADAM_LEARNING_RATE = 1e-4
PRINTING_INTERVAL = 10
# set seeds
np.random.seed(100)
tf.set_random_seed(100)
# set up training graph
x = tf.placeholder(tf.float32, shape=[None,
                                      DESIRED_SIZE,
                                      DESIRED_SIZE,
                                      3,
                                      NUMBER_OF_TRANSFORMATIONS])
y_gt = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])
keep_prob = tf.placeholder(tf.float32)
logits = rot_mnist12K_model.define_model(x,
                                         keep_prob,
                                         NUMBER_OF_CLASSES,
                                         NUMBER_OF_FILTERS,
                                         NUMBER_OF_FC_FEATURES)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= y_gt))
train_step = tf.train.AdamOptimizer(ADAM_LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run training
session = tf.Session()
session.run(tf.initialize_all_variables())
train_data_loader = toolstrain.DataLoader(TRAIN_FILENAME,
                                     NUMBER_OF_CLASSES,
                                     NUMBER_OF_TRANSFORMATIONS,
                                     LOADED_SIZE,
                                     DESIRED_SIZE)
test_data_loader = toolstrain.DataLoader(TEST_FILENAME,
                                    NUMBER_OF_CLASSES,
                                    NUMBER_OF_TRANSFORMATIONS,
                                    LOADED_SIZE,
                                    DESIRED_SIZE)
test_size = test_data_loader.all()[1].shape[0]
print "aa"
print test_size
print test_size % TEST_CHUNK_SIZE
assert test_size % TEST_CHUNK_SIZE == 0
number_of_test_chunks = test_size / TEST_CHUNK_SIZE
saver = tf.train.Saver()
while (True):
  batch = train_data_loader.next_batch(BATCH_SIZE)
  if (train_data_loader.is_new_epoch()):
    train_accuracy = session.run(accuracy, feed_dict={x : batch[0],
                                                      y_gt : batch[1],
                                                      keep_prob : 1.0})
    print("completed_epochs %d, training accuracy %g" %
          (train_data_loader.get_completed_epochs(), train_accuracy))
    saver.save(session, 'ckpt/tipooling.ckpt')
    sys.stdout.flush()
    if (train_data_loader.get_completed_epochs() % PRINTING_INTERVAL == 0):
      sum = 0.0
      for chunk_index in xrange(number_of_test_chunks):
        chunk = test_data_loader.next_batch(TEST_CHUNK_SIZE)
        sum += session.run(accuracy, feed_dict={x : chunk[0],
                                                y_gt : chunk[1],
                                                keep_prob : 1.0})
      test_accuracy = sum / number_of_test_chunks
      print("testing accuracy %g" % test_accuracy)
      sys.stdout.flush()
  session.run(train_step, feed_dict={x : batch[0],
                                     y_gt : batch[1],
                                     keep_prob : 0.5})


