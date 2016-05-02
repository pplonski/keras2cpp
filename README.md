# keras2cpp
This is a bunch of code to port Keras neural network model into pure C++.

## Usage
1. Save your Keras model architecture and weights:

    with open('my_nn_arch.json', 'w') as fout:
        fout.write(model.to_json())
    model.save_weights('my_nn_weights.h5')

2. Run converter, that will put architecture and weights into text file:

    python dump_to_simple_cpp.py -a my_nn_arch.json -w my_nn_weights.h5 -o nn_output.data

3. Use text file with network definition with KerasModel.cpp and KerasModel.h files. 

