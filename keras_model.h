#ifndef KERAS_MODEL__H
#define KERAS_MODEL__H

#include <string>
#include <vector>
#include <fstream>
class DataChunk {

};

class DataChunk2D : DataChunk {
public:
  std::vector<std::vector<std::vector<float> > > d; // depth, rows, cols
};

class DataChunkFlat : DataChunk {
public:
  std::vector<float> f;
};

class Layer {
public:
  virtual void load_weights(std::ifstream &fin) = 0;
  void compute_output();

  std::vector read_1d_array(std::ifstream &fin);
};


class LayerFlatten : public Layer {
public:
  void load_weights(std::ifstream &fin) {};
  void compute_output();
};


class LayerMaxPooling : public Layer {
public:
  void load_weights(std::ifstream &fin);
  void compute_output();

  int m_size_x;
  int m_size_y;
};

class LayerActivation : public Layer {
public:
  void load_weights(std::ifstream &fin);
  void compute_output();

  std::string m_activation_type;
};

class LayerConv2D : public Layer {
public:
  void load_weights(std::ifstream &fin);
  void compute_output();
  std::vector<std::vector<std::vector<std::vector<float> > > > kernels; // kernel, depth, rows, cols
  std::vector<float> bias; // kernel

  int m_kernels;
  int m_depth;
  int m_rows;
  int m_cols;
};

class LayerDense : Layer {
public:
  void load_weights(std::ifstream &fin);
  void compute_output();
  std::vector<std::vector<float> > w; //input, neuron
  std::vector<float> b; // neuron

  int m_input_cnt;
  int m_neurons;
};

class KerasModel {
public:
  KerasModel(const std::string &input_fname);
  std::vector<float> compute_output();
private:

  void load_weights(const std::string &input_fname);
  int m_layers_cnt; // number of layers
  std::vector<Layer *> m_layers; // container with layers

};

#endif
