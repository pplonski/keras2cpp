import numpy as np
np.random.seed(1336)
import json
import argparse
import sys

def get_numbers_from_file(fname):
    r = []
    with open(fname) as fin:
        for line in fin:
            tmp = line.split()
            r = [float(t) for t in tmp]
    return np.array(r)

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='This is a simple script compare predictions from keras and keras2cpp.')

parser.add_argument('-k', '--keras_response', help="Response from Keras (test_run_cnn.py)", required=True)
parser.add_argument('-c', '--keras2cpp_response', help="Response from Keras2cpp (test_run_cnn.cc)", required=True)
args = parser.parse_args()


keras_output     = get_numbers_from_file(args.keras_response)
keras2cpp_output = get_numbers_from_file(args.keras2cpp_response)

if len(keras_output) != len(keras2cpp_output):
    print "Different output dimensions"
    sys.exit(1)

sub = np.sum(np.abs(keras_output - keras2cpp_output))

if sub < 1e-6:
    print 'Test: [DONE]'
    print 'Dump is working correctly.'
    sys.exit(0)
else:
    print 'Test: [ERROR]'
    print 'The output from Keras and Keras2cpp are different.'
    print 'Difference value:', sub
    sys.exit(1)
