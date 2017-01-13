#include "keras_model.h"

#include <iostream>

using namespace std;
using namespace keras;



int main(int argc, char *argv[]) {

  if(argc != 4) {
    cout << "Wrong input, going to exit." << endl;
    cout << "There should be arguments: dumped_cnn_file input_sample output_file." << endl;
    return -1;
  }
  string dumped_cnn = argv[1];
  string input_data = argv[2];
  string response_file = argv[3];

  cout << "Testing network from " << dumped_cnn << " on data from " << input_data << endl;

  // Input data sample
  DataChunk *sample = new DataChunk2D();
  sample->read_from_file(input_data);

  // Construct network
  KerasModel m(dumped_cnn, false);
  std::vector<float> response = m.compute_output(sample);

  // clean sample
  delete sample;

  // save response into file
  ofstream fout(response_file);
  for(unsigned int i = 0; i < response.size(); i++) {
      fout << response[i] << " ";
  }
  fout.close();
  return 0;
}
