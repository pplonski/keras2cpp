# keras2cpp

This is a bunch of code to port Keras neural network model into pure C++. Neural network weights and architecture are stored in plain text file and input is presented as `vector<vector<vector<float> > >` in case of image. The code is prepared to support simple Convolutional network (from MNIST example) but can be easily extended. There are implemented only ReLU and Softmax activations.

It is working regardless the Keras backend.

#Usage

 1. Save your network weights and architecture.
 2. Dump network structure to plain text file with `dump_to_simple_cpp.py` script.
 3. Use network with code from `keras_model.h` and `keras_model.cc` files - see example below.

#Example

 1. Run one iteration of simple CNN on MNIST data with `example/mnist_cnn_one_iteration.py` script. It will produce files with architecture `example/my_nn_arch.json` and weights in HDF5 format `example/my_nn_weights.h5`.
 2. Dump network to plain text file `python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet`.
 3. Compile example `g++ -std=c++11 keras_model.cc example_main.cc` - see code in `example_main.cc`.
 4. Run binary `./a.out` - you shoul get the same output as in step one from Keras.
