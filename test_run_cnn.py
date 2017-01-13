import numpy as np
np.random.seed(1336)
from keras.models import Sequential, model_from_json
import json
import argparse
from keras import backend as K

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='This is a simple script to run Keras model from saved architecture and weights.\
                                              This script also creates a input data sample for c++ run_net.')

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-d', '--data_sample', help="File where to write random data sample", required=True)
parser.add_argument('-o', '--output', help="File where to write network outpu", required=True)
parser.add_argument('-v', '--verbose', help="Verbose", required=False)
args = parser.parse_args()

print 'Verbose', args.verbose
print 'Read architecture from', args.architecture
print 'Read weights from', args.weights

arch = open(args.architecture).read()
model = model_from_json(arch)
model.load_weights(args.weights)
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
arch = json.loads(arch)
print 'There are', str(len(model.layers)), 'layers in your network, is it good?'
print 'I think yes :)'


first_layer = arch["config"][0]["config"]
input_shape = first_layer["batch_input_shape"]
print "Input shape of your network", input_shape

print "Generate random input for testing purposes"
random_input = np.random.rand(1, input_shape[1], input_shape[2], input_shape[3])
print "Random input shape", random_input.shape
response = model.predict(random_input)[0]
if args.verbose:
    print '-'*50
    print 'Prediction from Keras'
    print response
    print '-'*50
# save response to the file
with open(args.output, "w") as fin:
    fin.write(' '.join([str(r) for r in response]))
# store one sample in text file
# this code is working for input_shape[1] == 1
if input_shape[1] != 1:
    print '-'*50
    print 'Sorry but below code can be a buggy for image depth > 1 !!!'
    print '-'*50

# save random data sample into file
with open(args.data_sample, "w") as fin:
    print "Save to", args.data_sample, "sample shape", str(input_shape[1]) + " " + str(input_shape[2]) + " " + str(input_shape[3])
    fin.write(str(input_shape[1]) + " " + str(input_shape[2]) + " " + str(input_shape[3]) + "\n")
    a = random_input[0,0]
    for b in a:
        fin.write(str(b)+'\n')

# Get layers output (for debuging)
'''
for l in xrange(len(model.layers)):
    with open('test_layer_' + str(l) + '.output', 'w') as fout:

        get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[l].output])
        layer_output = get_layer_output([random_input, 0])

        print 'Layer', l, layer_output[0].shape
        if l > 10:
            print layer_output
        fout.write(str(layer_output[0].shape) + '\n')
        fout.write(str(layer_output) + '\n')
'''
#print 'input?'
#get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                              [model.layers[0].input])
#layer_output = get_layer_output([random_input, 0])
#print layer_output
