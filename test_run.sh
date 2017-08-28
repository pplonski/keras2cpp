#!/bin/bash

echo 'Test for CNN dumping'

# Parameters
INPUT_ARCH="example/my_nn_arch.json"
INPUT_WEIGHTS="example/my_nn_weights.h5"

DUMPED_CNN="test_cnn.dumped"
DATA_SAMPLE="test_random_input.dat"
KERAS_OUTPUT="test_keras_output.dat"
KERAS2CPP_OUTPUT="test_keras2cpp_output.dat"
TEST_BIN="test_bin"

echo 'Test, step 1'
echo 'Dump network into plain text file' $DUMPED_CNN
python dump_to_simple_cpp.py -a $INPUT_ARCH -w $INPUT_WEIGHTS -o $DUMPED_CNN

echo 'Test, step 2'
echo 'Generate random input sample and save in' $DATA_SAMPLE
echo 'Compute ouput on generated sample with Keras and store predictions for comparison'
python test_run_cnn.py -a $INPUT_ARCH -w $INPUT_WEIGHTS -d $DATA_SAMPLE -o $KERAS_OUTPUT

echo 'Test, step 3'
echo 'Compile keras2cpp code'
g++ -std=c++11 test_run_cnn.cc keras_model.cc -o $TEST_BIN
echo 'Run predictions with dumped network and random data sample from step 2'
./$TEST_BIN $DUMPED_CNN $DATA_SAMPLE $KERAS2CPP_OUTPUT

echo 'Test, step 4'
echo 'Compare Keras and Keras2cpp outputs'
python test_compare.py --keras_response $KERAS_OUTPUT --keras2cpp_response $KERAS2CPP_OUTPUT

# Clean
echo 'Cleaning after test'
rm $DUMPED_CNN
rm $DATA_SAMPLE
rm $KERAS_OUTPUT
rm $KERAS2CPP_OUTPUT
rm $TEST_BIN
# used only if you log hidden layers output in test_run_cnn.py file
#rm test_layer_*.output
