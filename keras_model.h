#ifndef KERAS_MODEL__H
#define KERAS_MODEL__H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace keras
{
	std::vector<float> read_1d_array(std::ifstream &fin, int cols);
	void missing_activation_impl(const std::string &act);
	std::vector< std::vector<float> > conv_single_depth_valid(std::vector< std::vector<float> > const & im, std::vector< std::vector<float> > const & k);
	std::vector< std::vector<float> > conv_single_depth_same(std::vector< std::vector<float> > const & im, std::vector< std::vector<float> > const & k);

	class DataChunk;
	class DataChunk2D;
	class DataChunkFlat;

	class Layer;
	class LayerFlatten;
	class LayerMaxPooling;
	class LayerActivation;
	class LayerConv2D;
	class LayerDense;

	class KerasModel;
}

class keras::DataChunk {
public:
  virtual ~DataChunk() {}
  virtual size_t get_data_dim(void) const { return 0; }
  virtual std::vector<float> const & get_1d() const { throw "not implemented"; };
  virtual std::vector<std::vector<std::vector<float> > > const & get_3d() const { throw "not implemented"; };
  virtual void set_data(std::vector<std::vector<std::vector<float> > > const &) {};
  virtual void set_data(std::vector<float> const &) {};
  //virtual unsigned int get_count();
  virtual void read_from_file(const std::string &fname) {};
  virtual void show_name() = 0;
  virtual void show_values() = 0;
};

class keras::DataChunk2D : public keras::DataChunk {
public:
  std::vector< std::vector< std::vector<float> > > const & get_3d() const { return data; };
  virtual void set_data(std::vector<std::vector<std::vector<float> > > const & d) { data = d; };
  size_t get_data_dim(void) const { return 3; }

  void show_name() {
    std::cout << "DataChunk2D " << data.size() << "x" << data[0].size() << "x" << data[0][0].size() << std::endl;
  }

  void show_values() {
    std::cout << "DataChunk2D values:" << std::endl;
    for(size_t i = 0; i < data.size(); ++i) {
      std::cout << "Kernel " << i << std::endl;
      for(size_t j = 0; j < data[0].size(); ++j) {
        for(size_t k = 0; k < data[0][0].size(); ++k) {
          std::cout << data[i][j][k] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
  //unsigned int get_count() {
  //  return data.size()*data[0].size()*data[0][0].size();
  //}

  void read_from_file(const std::string &fname);
  std::vector<std::vector<std::vector<float> > > data; // depth, rows, cols

  int m_depth;
  int m_rows;
  int m_cols;
};

class keras::DataChunkFlat : public keras::DataChunk {
public:
  DataChunkFlat(size_t size) : f(size) { }
  DataChunkFlat(size_t size, float init) : f(size, init) { }
  DataChunkFlat(void) { }

  std::vector<float> f;
  std::vector<float> & get_1d_rw() { return f; }
  std::vector<float> const & get_1d() const { return f; }
  void set_data(std::vector<float> const & d) { f = d; };
  size_t get_data_dim(void) const { return 1; }

  void show_name() {
    std::cout << "DataChunkFlat " << f.size() << std::endl;
  }
  void show_values() {
    std::cout << "DataChunkFlat values:" << std::endl;
    for(size_t i = 0; i < f.size(); ++i) std::cout << f[i] << " ";
    std::cout << std::endl;
  }
  void read_from_file(const std::string &fname) {};
  //unsigned int get_count() { return f.size(); }
};

class keras::Layer {
public:
  virtual void load_weights(std::ifstream &fin) = 0;
  virtual keras::DataChunk* compute_output(keras::DataChunk*) = 0;

  Layer(std::string name) : m_name(name) {}
  virtual ~Layer() {}

  virtual unsigned int get_input_rows() const = 0;
  virtual unsigned int get_input_cols() const = 0;
  virtual unsigned int get_output_units() const = 0;

  std::string get_name() { return m_name; }
  std::string m_name;
};


class keras::LayerFlatten : public Layer {
public:
  LayerFlatten() : Layer("Flatten") {}
  void load_weights(std::ifstream &fin) {};
  keras::DataChunk* compute_output(keras::DataChunk*);

  virtual unsigned int get_input_rows() const { return 0; } // look for the value in the preceding layer
  virtual unsigned int get_input_cols() const { return 0; } // same as for rows
  virtual unsigned int get_output_units() const { return 0; }
};


class keras::LayerMaxPooling : public Layer {
public:
  LayerMaxPooling() : Layer("MaxPooling2D") {};

  void load_weights(std::ifstream &fin);
  keras::DataChunk* compute_output(keras::DataChunk*);

  virtual unsigned int get_input_rows() const { return 0; } // look for the value in the preceding layer
  virtual unsigned int get_input_cols() const { return 0; } // same as for rows
  virtual unsigned int get_output_units() const { return 0; }

  int m_pool_x;
  int m_pool_y;

};

class keras::LayerActivation : public Layer {
public:
  LayerActivation() : Layer("Activation") {}
  void load_weights(std::ifstream &fin);
  keras::DataChunk* compute_output(keras::DataChunk*);

  virtual unsigned int get_input_rows() const { return 0; } // look for the value in the preceding layer
  virtual unsigned int get_input_cols() const { return 0; } // same as for rows
  virtual unsigned int get_output_units() const { return 0; }

  std::string m_activation_type;
};

class keras::LayerConv2D : public Layer {
public:
  LayerConv2D() : Layer("Conv2D") {}

  void load_weights(std::ifstream &fin);
  keras::DataChunk* compute_output(keras::DataChunk*);
  std::vector<std::vector<std::vector<std::vector<float> > > > m_kernels; // kernel, depth, rows, cols
  std::vector<float> m_bias; // kernel

  virtual unsigned int get_input_rows() const { return m_rows; }
  virtual unsigned int get_input_cols() const { return m_cols; }
  virtual unsigned int get_output_units() const { return m_kernels_cnt; }

  std::string m_border_mode;
  int m_kernels_cnt;
  int m_depth;
  int m_rows;
  int m_cols;
};

class keras::LayerDense : public Layer {
public:
  LayerDense() : Layer("Dense") {}

  void load_weights(std::ifstream &fin);
  keras::DataChunk* compute_output(keras::DataChunk*);
  std::vector<std::vector<float> > m_weights; //input, neuron
  std::vector<float> m_bias; // neuron

  virtual unsigned int get_input_rows() const { return 1; } // flat, just one row
  virtual unsigned int get_input_cols() const { return m_input_cnt; }
  virtual unsigned int get_output_units() const { return m_neurons; }

  int m_input_cnt;
  int m_neurons;
};

class keras::KerasModel {
public:
  KerasModel(const std::string &input_fname, bool verbose);
  ~KerasModel();
  std::vector<float> compute_output(keras::DataChunk *dc);

  unsigned int get_input_rows() const { return m_layers.front()->get_input_rows(); }
  unsigned int get_input_cols() const { return m_layers.front()->get_input_cols(); }
  int get_output_length() const;

private:

  void load_weights(const std::string &input_fname);
  int m_layers_cnt; // number of layers
  std::vector<Layer *> m_layers; // container with layers
  bool m_verbose;

};

#endif
