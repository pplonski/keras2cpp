#include "keras_model.h"

#include <iostream>

using namespace std;
using namespace keras;

// Step 1
// Dump keras model and input sample into text files
// python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet
// Step 2
// Use text files in c++ example. To compile:
// g++ keras_model.cc example_main.cc
// To execute:
// a.out

int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\n"
           << "Keras model will be used in C++ for prediction only." << endl;

  DataChunk *sample = new DataChunk2D();
  sample->read_from_file("./example/sample_mnist.dat");
  std::cout << sample->get_3d().size() << std::endl;
  KerasModel m("./example/dumped.nnet", true);
  m.compute_output(sample);
  delete sample;

  return 0;
}
