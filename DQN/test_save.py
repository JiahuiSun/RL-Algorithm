# coding: utf-8
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

lr = 0.001
n_input = 28 * 28
n_class = 10

mnist = input_data.read_data_sets('./', one_hot=True)