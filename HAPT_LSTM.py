# Filename: HAPT_LSTM.py
# Dependencies: HAPT_LSTM_UTIL (keras, numpy, os, pandas, sklearn, tensorflow)
# Author: Jean-Michel Boudreau
# Date: June 5, 2019

'''
Loads Human Activities and Postural Transitions data set (must be 
colocated in directory with this script) and trains a Long-Short Term Memory 
RNN to classify the motion of individuals within the data set.
'''

from HAPT_LSTM_UTIL import train_LSTM, test_LSTM
# Train LSTM network
train_LSTM()
# Test LSTM network
test_LSTM()
