#ifndef KERAS_MODEL__H
#define KERAS_MODEL__H

#include <string>
#include <vector>
#include <fstream>

class DataChunk {
public:
  virtual std::vector<float> get_1d() {
    return std::vector<float>();
  };
  virtual std::vector<std::vector<std::vector<float> > > get_3d() {
    return   std::vector<std::vector<std::vector<float> > > ();
  };
  virtual void read_from_file(const std::string &fname) {};
};

class DataChunk2D : public DataChunk {
public:
  // return empty vector

  std::vector<std::vector<std::vector<float> > > get_3d() {
    return data;
  };

  void read_from_file(const std::string &fname);
  std::vector<std::vector<std::vector<float> > > data; // depth, rows, cols

  int m_depth;
  int m_rows;
  int m_cols;
};

class DataChunkFlat : public DataChunk {
public:
  void read_from_file(const std::string &fname) {};
  std::vector<float> f;
  std::vector<float> get_1d() {
    return f;
  };

};

class Layer {
public:
  virtual void load_weights(std::ifstream &fin) = 0;
  void compute_output();

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

class LayerDense : public Layer {
public:
  void load_weights(std::ifstream &fin);
  void compute_output();
  std::vector<std::vector<float> > weights; //input, neuron
  std::vector<float> bias; // neuron

  int m_input_cnt;
  int m_neurons;
};

class KerasModel {
public:
  KerasModel(const std::string &input_fname);
  ~KerasModel();
  std::vector<float> compute_output(DataChunk *dc);
private:

  void load_weights(const std::string &input_fname);
  int m_layers_cnt; // number of layers
  std::vector<Layer *> m_layers; // container with layers

};

#endif
