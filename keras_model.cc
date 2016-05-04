#include "keras_model.h"

#include <iostream>
#include <fstream>

using namespace std;


std::vector<float> read_1d_array(std::ifstream &fin, int cols) {
  vector<float> arr;
  float tmp_float;
  float tmp_char;
  fin >> tmp_char; // for '['
  for(int n = 0; n < cols; ++n) {
    fin >> tmp_float;
    arr.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'
  return arr;
}

void DataChunk2D::read_from_file(const std::string &fname) {
  ifstream fin(fname);
  fin >> m_depth >> m_rows >> m_cols;
  for(int d = 0; d < m_depth; ++d) {
    vector<vector<float> > tmp_single_depth;
    for(int r = 0; r < m_rows; ++r) {
      vector<float> tmp_row = read_1d_array(fin, m_cols);
      tmp_single_depth.push_back(tmp_row);
    }
    data.push_back(tmp_single_depth);
  }
  fin.close();
  cout << "data " << data.size() << "x" << data[0].size() << "x" << data[0][0].size() << endl;
}


void LayerConv2D::load_weights(std::ifstream &fin) {
  char tmp_char = ' ';
  string tmp_str = "";
  int tmp_int = 0;
  float tmp_float;
  fin >> m_kernels >> m_depth >> m_rows >> m_cols;
  cout << "LayerConv2D " << m_kernels << "x" << m_depth << "x" << m_rows << "x" << m_cols << endl;
  // reading kernel weights
  for(int k = 0; k < m_kernels; ++k) {
    vector<vector<vector<float> > > tmp_depths;
    for(int d = 0; d < m_depth; ++d) {
      vector<vector<float> > tmp_single_depth;
      for(int r = 0; r < m_rows; ++r) {
        fin >> tmp_char; // for '['
        vector<float> tmp_row;
        for(int c = 0; c < m_cols; ++c) {
          fin >> tmp_float;
          tmp_row.push_back(tmp_float);
        }
        fin >> tmp_char; // for ']'
        tmp_single_depth.push_back(tmp_row);
      }
      tmp_depths.push_back(tmp_single_depth);
    }
    kernels.push_back(tmp_depths);
  }
  /*
  for(int i = 0; i < kernels.size(); ++i) {
    cout << i << " " << kernels[i].size() << endl;
    for(int j = 0; j < kernels[i].size(); ++j) {
      cout << j << " " << kernels[i][j].size() << endl;
      for(int k = 0; k < kernels[i][j].size(); ++k) {
        cout << k << " " << kernels[i][j][k].size() << endl;
      }
    }
  }*/
  // reading kernel biases
  fin >> tmp_char; // for '['
  for(int k = 0; k < m_kernels; ++k) {
    fin >> tmp_float;
    bias.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'

  /*for(int k = 0; k < m_kernels; ++k) {
    cout << bias[k] << " ";
  }
  cout << endl;*/
}

void LayerActivation::load_weights(std::ifstream &fin) {
  fin >> m_activation_type;
  cout << "Activation type " << m_activation_type << endl;
}

void LayerMaxPooling::load_weights(std::ifstream &fin) {
  fin >> m_size_x >> m_size_y;
  cout << "MaxPooling " << m_size_x << "x" << m_size_y << endl;
}

void LayerDense::load_weights(std::ifstream &fin) {
  fin >> m_input_cnt >> m_neurons;
  float tmp_float;
  char tmp_char = ' ';
  for(int i = 0; i < m_input_cnt; ++i) {
    vector<float> tmp_n;
    fin >> tmp_char; // for '['
    for(int n = 0; n < m_neurons; ++n) {
      fin >> tmp_float;
      tmp_n.push_back(tmp_float);
    }
    fin >> tmp_char; // for ']'
    weights.push_back(tmp_n);
  }
  cout << "weights " << weights.size() << endl;
  fin >> tmp_char; // for '['
  for(int n = 0; n < m_neurons; ++n) {
    fin >> tmp_float;
    bias.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'
  cout << "bias " << bias.size() << endl;

}

KerasModel::KerasModel(const string &input_fname) {
  cout << "KerasModel ctor" << endl;
  load_weights(input_fname);
}



void KerasModel::load_weights(const string &input_fname) {
  cout << "Reading model from " << input_fname << endl;
  ifstream fin(input_fname);
  string layer_type = "";
  char tmp_char = ' ';
  string tmp_str = "";
  int tmp_int = 0;
  float tmp_float = 0;

  fin >> tmp_str >> m_layers_cnt;
  cout << "Layers " << m_layers_cnt << endl;

  for(int layer = 0; layer < m_layers_cnt; ++layer) { // iterate over layers
    fin >> tmp_str >> tmp_int >> layer_type;
    cout << "Layer " << tmp_int << " " << layer_type << endl;

    Layer *l = 0L;
    if(layer_type == "Convolution2D") {
      l = new LayerConv2D();
    } else if(layer_type == "Activation") {
      l = new LayerActivation();
    } else if(layer_type == "MaxPooling2D") {
      l = new LayerMaxPooling();
    } else if(layer_type == "Flatten") {
      l = new LayerFlatten();
    } else if(layer_type == "Dense") {
      l = new LayerDense();
    } else if(layer_type == "Dropout") {
      continue; // we dont need dropout layer in prediciton mode
    }
    if(l == 0L) {
      cout << "Layer is empty, maybe it is not defined? Cannot define network." << endl;
      return;
    }
    l->load_weights(fin);

    //if(layer > 3) break;
  }

  fin.close();
}
