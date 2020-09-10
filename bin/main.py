import sys,os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)



import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from core import lstm2
if __name__ == '__main__':

    # t1=tf.constant([[1],[2]])
    # print(t1.shape)
    # f=Factor([0,1,2,3,4,5,6,7],0,1,2,3,4,5,6,7)
    # print(f)

    # from lib import common
    # import time
    #
    # logger=common.get_logger(__name__)
    # logger.error('cccccc')

    from keras.datasets import mnist
    print(2)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_labels.shape)


