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
  ifstream fin(fname.c_str());
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
  load_weights(input_fname);
}


DataChunk* LayerFlatten::compute_output(DataChunk* dc) {
  return dc;
}
DataChunk* LayerMaxPooling::compute_output(DataChunk* dc) {
  return dc;
}
DataChunk* LayerActivation::compute_output(DataChunk* dc) {
  return dc;
}


def my_conv(im, k):
    st_x = int((k.shape[0]-1)/2.0)
    st_y = int((k.shape[1]-1)/2.0)

    y = np.zeros((im.shape[0]-2*st_x, im.shape[1]-2*st_y))
    for i in range(st_x, int(im.shape[0]-st_x)):
        for j in range(st_y, int(im.shape[0]-st_y)):
            for k1 in range(0,k.shape[0]):
                for k2 in range(0,k.shape[1]):
                    y[i-st_x,j-st_y] += k[k.shape[0]-k1-1][k.shape[1]-k2-1] * im[i-st_x+k1, j-st_y+k2]
    return y

vector<vector<float> > conv_single_depth(vector<vector<float> > im, vector<vector<float> > k) {

}

DataChunk* LayerConv2D::compute_output(DataChunk* dc) {
  return dc;
}
DataChunk* LayerDense::compute_output(DataChunk* dc) {
  return dc;
}


std::vector<float> KerasModel::compute_output(DataChunk *dc) {
  cout << "KerasModel compute output" << endl;
  cout << dc->get_3d().size() << endl;

  //DataChunk *tmp;
  DataChunk *inp = dc;
  DataChunk *out;
  for(int l = 0; l < (int)m_layers.size(); ++l) {
    cout << "-----------------\nOuput from layer " << m_layers[l]->get_name() << endl;
    out = m_layers[l]->compute_output(inp);

    cout << "Input" << endl;
    inp->show_name();
    cout << "Output" << endl;
    out->show_name();
    //tmp = out;
    // delete inp
    inp = out;

    break;
  }


  vector<float> r;
  return r;
}

void KerasModel::load_weights(const string &input_fname) {
  cout << "Reading model from " << input_fname << endl;
  ifstream fin(input_fname.c_str());
  string layer_type = "";
  string tmp_str = "";
  int tmp_int = 0;

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
    m_layers.push_back(l);
    //if(layer > 3) break;
  }

  fin.close();
}

KerasModel::~KerasModel() {
  for(int i = 0; i < (int)m_layers.size(); ++i) {
    delete m_layers[i];
  }
}
