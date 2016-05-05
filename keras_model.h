#ifndef KERAS_MODEL__H
#define KERAS_MODEL__H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class DataChunk {
public:
  virtual ~DataChunk() {}
  virtual std::vector<float> get_1d() {
    return std::vector<float>();
  };
  virtual std::vector<std::vector<std::vector<float> > > get_3d() {
    return   std::vector<std::vector<std::vector<float> > > ();
  };

  virtual void set_data(std::vector<std::vector<std::vector<float> > > ) {};
  virtual void set_data(std::vector<float> ) {};

  virtual void read_from_file(const std::string &fname) {};
  virtual void show_name() = 0;
};

class DataChunk2D : public DataChunk {
public:
  std::vector<std::vector<std::vector<float> > > get_3d() {
    return data;
  };
  virtual void set_data(std::vector<std::vector<std::vector<float> > > d) { data = d; };

  void show_name() {
    std::cout << "DataChunk2D " << data.size() << "x" << data[0].size() << "x" << data[0][0].size() << std::endl;
  }

  void read_from_file(const std::string &fname);
  std::vector<std::vector<std::vector<float> > > data; // depth, rows, cols

  int m_depth;
  int m_rows;
  int m_cols;
};

class DataChunkFlat : public DataChunk {
public:
  void set_data(std::vector<float> d) { f = d; };

  void show_name() {
    std::cout << "DataChunkFlat " << f.size() << std::endl;
  }
  void read_from_file(const std::string &fname) {};
  std::vector<float> f;
  std::vector<float> get_1d() {
    return f;
  };

};

class Layer {
public:
  virtual void load_weights(std::ifstream &fin) = 0;
  virtual DataChunk* compute_output(DataChunk*) = 0;

  Layer(std::string name) : m_name(name) {}
  virtual ~Layer() {}
  std::string get_name() { return m_name; }
  std::string m_name;
};


class LayerFlatten : public Layer {
public:
  LayerFlatten() : Layer("Flatten") {}
  void load_weights(std::ifstream &fin) {};
  DataChunk* compute_output(DataChunk*);

};


class LayerMaxPooling : public Layer {
public:
  LayerMaxPooling() : Layer("MaxPooling2D") {};

  void load_weights(std::ifstream &fin);
  DataChunk* compute_output(DataChunk*);

  int m_size_x;
  int m_size_y;

};

class LayerActivation : public Layer {
public:
  LayerActivation() : Layer("Activation") {}
  void load_weights(std::ifstream &fin);
  DataChunk* compute_output(DataChunk*);

  std::string m_activation_type;
};

class LayerConv2D : public Layer {
public:
  LayerConv2D() : Layer("Conv2D") {}

  void load_weights(std::ifstream &fin);
  DataChunk* compute_output(DataChunk*);
  std::vector<std::vector<std::vector<std::vector<float> > > > m_kernels; // kernel, depth, rows, cols
  std::vector<float> m_bias; // kernel

  int m_kernels_cnt;
  int m_depth;
  int m_rows;
  int m_cols;
};

class LayerDense : public Layer {
public:
  LayerDense() : Layer("Dense") {}

  void load_weights(std::ifstream &fin);
  DataChunk* compute_output(DataChunk*);
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
