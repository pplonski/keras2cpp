#include "keras_model.h"

#include <iostream>

using namespace std;

int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\nKeras model will be used in C++ for prediction only." << endl;

  DataChunk *sample = new DataChunk2D();
  sample->read_from_file("./sample_mnist.dat");
  std::cout << sample->get_3d().size() << std::endl;
  KerasModel m("nnet.dat");
  m.compute_output(sample);

  //delete sample;
  return 0;
}
