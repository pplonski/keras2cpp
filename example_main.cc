#include "keras_model.h"

#include <iostream>

using namespace std;

//python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet
//g++ keras_model.cc example_main.cc
int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\nKeras model will be used in C++ for prediction only." << endl;

  DataChunk *sample = new DataChunk2D();
  sample->read_from_file("./example/sample_mnist.dat");
  std::cout << sample->get_3d().size() << std::endl;
  KerasModel m("./example/dumped.nnet");
  m.compute_output(sample);

  return 0;
}
