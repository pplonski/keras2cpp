#include "keras_model.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
using namespace std;


std::vector<float> keras::read_1d_array(std::ifstream &fin, int cols) {
  vector<float> arr;
  float tmp_float;
  char tmp_char;
  fin >> tmp_char; // for '['
  for(int n = 0; n < cols; ++n) {
    fin >> tmp_float;
    arr.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'
  return arr;
}

void keras::DataChunk2D::read_from_file(const std::string &fname) {
  ifstream fin(fname.c_str());
  fin >> m_depth >> m_rows >> m_cols;

  for(int d = 0; d < m_depth; ++d) {
    vector<vector<float> > tmp_single_depth;
    for(int r = 0; r < m_rows; ++r) {
      vector<float> tmp_row = keras::read_1d_array(fin, m_cols);
      tmp_single_depth.push_back(tmp_row);
    }
    data.push_back(tmp_single_depth);
  }
  fin.close();
}


void keras::LayerConv2D::load_weights(std::ifstream &fin) {
  char tmp_char = ' ';
  string tmp_str = "";
  float tmp_float;
  bool skip = false;
  fin >> m_kernels_cnt >> m_depth >> m_rows >> m_cols >> m_border_mode;
  if (m_border_mode == "[") { m_border_mode = "valid"; skip = true; }

  //cout << "LayerConv2D " << m_kernels_cnt << "x" << m_depth << "x" << m_rows <<
  //            "x" << m_cols << " border_mode " << m_border_mode << endl;
  // reading kernel weights
  for(int k = 0; k < m_kernels_cnt; ++k) {
    vector<vector<vector<float> > > tmp_depths;
    for(int d = 0; d < m_depth; ++d) {
      vector<vector<float> > tmp_single_depth;
      for(int r = 0; r < m_rows; ++r) {
        if (!skip) { fin >> tmp_char; } // for '['
        else { skip = false; }
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
    m_kernels.push_back(tmp_depths);
  }
  // reading kernel biases
  fin >> tmp_char; // for '['
  for(int k = 0; k < m_kernels_cnt; ++k) {
    fin >> tmp_float;
    m_bias.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'

}

void keras::LayerActivation::load_weights(std::ifstream &fin) {
  fin >> m_activation_type;
  //cout << "Activation type " << m_activation_type << endl;
}

void keras::LayerMaxPooling::load_weights(std::ifstream &fin) {
  fin >> m_pool_x >> m_pool_y;
  //cout << "MaxPooling " << m_pool_x << "x" << m_pool_y << endl;
}

void keras::LayerDense::load_weights(std::ifstream &fin) {
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
    m_weights.push_back(tmp_n);
  }
  //cout << "weights " << m_weights.size() << endl;
  fin >> tmp_char; // for '['
  for(int n = 0; n < m_neurons; ++n) {
    fin >> tmp_float;
    m_bias.push_back(tmp_float);
  }
  fin >> tmp_char; // for ']'
  //cout << "bias " << m_bias.size() << endl;

}

keras::KerasModel::KerasModel(const string &input_fname, bool verbose)
                                                       : m_verbose(verbose) {
  load_weights(input_fname);
}


keras::DataChunk* keras::LayerFlatten::compute_output(keras::DataChunk* dc) {
  vector<vector<vector<float> > > im = dc->get_3d();

  size_t csize = im[0].size();
  size_t rsize = im[0][0].size();
  size_t size = im.size() * csize * rsize;
  keras::DataChunkFlat *out = new DataChunkFlat(size);
  float * y_ret = out->get_1d_rw().data();
  for(size_t i = 0, dst = 0; i < im.size(); ++i) {
    for(size_t j = 0; j < csize; ++j) {
      float * row = im[i][j].data();
      for(size_t k = 0; k < rsize; ++k) {
        y_ret[dst++] = row[k];
      }
    }
  }

  return out;
}


keras::DataChunk* keras::LayerMaxPooling::compute_output(keras::DataChunk* dc) {
  vector<vector<vector<float> > > im = dc->get_3d();
  vector<vector<vector<float> > > y_ret;
  for(unsigned int i = 0; i < im.size(); ++i) {
    vector<vector<float> > tmp_y;
    for(unsigned int j = 0; j < (unsigned int)(im[0].size()/m_pool_x); ++j) {
      tmp_y.push_back(vector<float>((int)(im[0][0].size()/m_pool_y), 0.0));
    }
    y_ret.push_back(tmp_y);
  }
  for(unsigned int d = 0; d < y_ret.size(); ++d) {
    for(unsigned int x = 0; x < y_ret[0].size(); ++x) {
      unsigned int start_x = x*m_pool_x;
      unsigned int end_x = start_x + m_pool_x;
      for(unsigned int y = 0; y < y_ret[0][0].size(); ++y) {
        unsigned int start_y = y*m_pool_y;
        unsigned int end_y = start_y + m_pool_y;

        vector<float> values;
        for(unsigned int i = start_x; i < end_x; ++i) {
          for(unsigned int j = start_y; j < end_y; ++j) {
            values.push_back(im[d][i][j]);
          }
        }
        y_ret[d][x][y] = *max_element(values.begin(), values.end());
      }
    }
  }
  keras::DataChunk *out = new keras::DataChunk2D();
  out->set_data(y_ret);
  return out;
}

void keras::missing_activation_impl(const string &act) {
  cout << "Activation " << act << " not defined!" << endl;
  cout << "Please add its implementation before use." << endl;
  exit(1);
}

keras::DataChunk* keras::LayerActivation::compute_output(keras::DataChunk* dc) {

  if (dc->get_data_dim() == 3) {
    vector<vector<vector<float> > > y = dc->get_3d();
    if(m_activation_type == "relu") {
      for(unsigned int i = 0; i < y.size(); ++i) {
        for(unsigned int j = 0; j < y[0].size(); ++j) {
          for(unsigned int k = 0; k < y[0][0].size(); ++k) {
            if(y[i][j][k] < 0) y[i][j][k] = 0;
          }
        }
      }
      keras::DataChunk *out = new keras::DataChunk2D();
      out->set_data(y);
      return out;
    } else {
      keras::missing_activation_impl(m_activation_type);
    }
  } else if (dc->get_data_dim() == 1) { // flat data, use 1D
    vector<float> y = dc->get_1d();
    if(m_activation_type == "relu") {
      for(unsigned int k = 0; k < y.size(); ++k) {
        if(y[k] < 0) y[k] = 0;
      }
    } else if(m_activation_type == "softmax") {
      float sum = 0.0;
      for(unsigned int k = 0; k < y.size(); ++k) {
        y[k] = exp(y[k]);
        sum += y[k];
      }
      for(unsigned int k = 0; k < y.size(); ++k) {
        y[k] /= sum;
      }
    } else if(m_activation_type == "sigmoid") {
      for(unsigned int k = 0; k < y.size(); ++k) {
        y[k] = 1/(1+exp(-y[k]));
      }
    } else if(m_activation_type == "tanh") {
      for(unsigned int k = 0; k < y.size(); ++k) {
        y[k] = tanh(y[k]);
      }
    } else {
      keras::missing_activation_impl(m_activation_type);
    }

    keras::DataChunk *out = new DataChunkFlat();
    out->set_data(y);
    return out;
  } else { throw "data dim not supported"; }

  return dc;
}


// with border mode = valid
std::vector< std::vector<float> > keras::conv_single_depth_valid(
	std::vector< std::vector<float> > const & im,
	std::vector< std::vector<float> > const & k)
{
  size_t k1_size = k.size(), k2_size = k[0].size();
  unsigned int st_x = (k1_size - 1) >> 1;
  unsigned int st_y = (k2_size - 1) >> 1;

  std::vector< std::vector<float> > y(im.size() - 2*st_x, vector<float>(im[0].size() - 2*st_y, 0));

  for(unsigned int i = st_x; i < im.size()-st_x; ++i) {
    for(unsigned int j = st_y; j < im[0].size()-st_y; ++j) {

      float sum = 0;
      for(unsigned int k1 = 0; k1 < k.size(); ++k1) {
        //const float * k_data = k[k1_size-k1-1].data();
        //const float * im_data = im[i-st_x+k1].data();
        for(unsigned int k2 = 0; k2 < k[0].size(); ++k2) {
          sum += k[k1_size-k1-1][k2_size-k2-1] * im[i-st_x+k1][j-st_y+k2];
        }
      }
      y[i-st_x][j-st_y] = sum;
    }
  }
  return y;
}


// with border mode = same
std::vector< std::vector<float> > keras::conv_single_depth_same(
	std::vector< std::vector<float> > const & im,
	std::vector< std::vector<float> > const & k)
{
  size_t k1_size = k.size(), k2_size = k[0].size();
  unsigned int st_x = (k1_size - 1) >> 1;
  unsigned int st_y = (k2_size - 1) >> 1;

  size_t max_imc = im.size() - 1;
  size_t max_imr = im[0].size() - 1;
  std::vector< std::vector<float> > y(im.size(), vector<float>(im[0].size(), 0));

  for(unsigned int i = 0; i < im.size(); ++i) {
    for(unsigned int j = 0; j < im[0].size(); ++j) {
      float sum = 0;
      for(unsigned int k1 = 0; k1 < k.size(); ++k1) {
        //const float * k_data = k[k1_size-k1-1].data(); // it is not working ...
        //const float * im_data = im[i-st_x+k1].data();
        for(unsigned int k2 = 0; k2 < k[0].size(); ++k2) {
          if(i-st_x+k1 < 0) continue;
          if(i-st_x+k1 > max_imc) continue;
          if(j-st_y+k2 < 0) continue;
          if(j-st_y+k2 > max_imr) continue;

          sum += k[k1_size-k1-1][k2_size-k2-1] * im[i-st_x+k1][j-st_y+k2];
        }
      }
      y[i][j] = sum;
    }
  }
  return y;
}


keras::DataChunk* keras::LayerConv2D::compute_output(keras::DataChunk* dc) {

  unsigned int st_x = (m_kernels[0][0].size()-1) >> 1;
  unsigned int st_y = (m_kernels[0][0][0].size()-1) >> 1;
  vector< vector< vector<float> > > y_ret;
  auto const & im = dc->get_3d();

  size_t size_x = (m_border_mode == "valid")? im[0].size() - 2 * st_x : im[0].size();
  size_t size_y = (m_border_mode == "valid")? im[0][0].size() - 2 * st_y: im[0][0].size();
  for(unsigned int i = 0; i < m_kernels.size(); ++i) { // depth
    vector<vector<float> > tmp;
    tmp.reserve(size_x);
    for(unsigned int j = 0; j < size_x; ++j) { // rows
      tmp.emplace_back(vector<float>(size_y, 0.0));
    }
    y_ret.push_back(tmp);
  }

  for(unsigned int j = 0; j < m_kernels.size(); ++j) { // loop over kernels
    for(unsigned int m = 0; m < im.size(); ++m) { // loope over image depth

      vector<vector<float> > tmp_w = (m_border_mode == "valid")?
                        keras::conv_single_depth_valid(im[m], m_kernels[j][m]) :
                        keras::conv_single_depth_same(im[m], m_kernels[j][m]);

      for(unsigned int x = 0; x < tmp_w.size(); ++x) {
        for(unsigned int y = 0; y < tmp_w[0].size(); ++y) {
          y_ret[j][x][y] += tmp_w[x][y];
        }
      }
    }

    for(unsigned int x = 0; x < y_ret[0].size(); ++x) {
      for(unsigned int y = 0; y < y_ret[0][0].size(); ++y) {
        y_ret[j][x][y] += m_bias[j];
      }
    }
  }

  keras::DataChunk *out = new keras::DataChunk2D();
  out->set_data(y_ret);
  return out;
}

keras::DataChunk* keras::LayerDense::compute_output(keras::DataChunk* dc) {
  //cout << "weights: input size " << m_weights.size() << endl;
  //cout << "weights: neurons size " << m_weights[0].size() << endl;
  //cout << "bias " << m_bias.size() << endl;
  size_t size = m_weights[0].size();
  size_t size8 = size >> 3;
  keras::DataChunkFlat *out = new DataChunkFlat(size, 0);
  float * y_ret = out->get_1d_rw().data();

  auto const & im = dc->get_1d();

  for (size_t j = 0; j < m_weights.size(); ++j) { // iter over input
    const float * w = m_weights[j].data();
    float p = im[j];
    size_t k = 0;
    for (size_t i = 0; i < size8; ++i) { // iter over neurons
      y_ret[k]   += w[k]   * p;          // vectorize if you can
      y_ret[k+1] += w[k+1] * p;
      y_ret[k+2] += w[k+2] * p;
      y_ret[k+3] += w[k+3] * p;
      y_ret[k+4] += w[k+4] * p;
      y_ret[k+5] += w[k+5] * p;
      y_ret[k+6] += w[k+6] * p;
      y_ret[k+7] += w[k+7] * p;
      k += 8;
    }
    while (k < size) { y_ret[k] += w[k] * p; ++k; }
  }
  for (size_t i = 0; i < size; ++i) { // add biases
    y_ret[i] += m_bias[i];
  }

  return out;
}


std::vector<float> keras::KerasModel::compute_output(keras::DataChunk *dc) {
  //cout << endl << "KerasModel compute output" << endl;
  //cout << "Input data size:" << endl;
  dc->show_name();

  keras::DataChunk *inp = dc;
  keras::DataChunk *out = 0;
  for(int l = 0; l < (int)m_layers.size(); ++l) {
    //cout << "Processing layer " << m_layers[l]->get_name() << endl;
    out = m_layers[l]->compute_output(inp);

    //cout << "Input" << endl;
    //inp->show_name();
    //cout << "Output" << endl;
    //out->show_name();
    if(inp != dc) delete inp;
    //delete inp;
    inp = 0L;
    inp = out;
  }

  std::vector<float> flat_out = out->get_1d();
  out->show_values();
  delete out;

  return flat_out;
}

void keras::KerasModel::load_weights(const string &input_fname) {
  if(m_verbose) cout << "Reading model from " << input_fname << endl;
  ifstream fin(input_fname.c_str());
  string layer_type = "";
  string tmp_str = "";
  int tmp_int = 0;

  fin >> tmp_str >> m_layers_cnt;
  if(m_verbose) cout << "Layers " << m_layers_cnt << endl;

  for(int layer = 0; layer < m_layers_cnt; ++layer) { // iterate over layers
    fin >> tmp_str >> tmp_int >> layer_type;
    if(m_verbose) cout << "Layer " << tmp_int << " " << layer_type << endl;

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
  }

  fin.close();
}

keras::KerasModel::~KerasModel() {
  for(int i = 0; i < (int)m_layers.size(); ++i) {
    delete m_layers[i];
  }
}

int keras::KerasModel::get_output_length() const
{
  int i = m_layers.size() - 1;
  while ((i > 0) && (m_layers[i]->get_output_units() == 0)) --i;
  return m_layers[i]->get_output_units();
}
