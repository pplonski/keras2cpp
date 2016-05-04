#include "keras_model.h"

#include <iostream>

using namespace std;

int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\nKeras model will be used in C++ for prediction only." << endl;
  KerasModel m("nn.dat");

  return 0;
}
