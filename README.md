# TensorFlow-TextGCN
Efficient implementation of "Graph Convolutional Networks for Text Classification"

## Requirements

+ Python >= 3.5
+ tqdm: `pip install tqdm`
+ scikit-learn: `pip install scikit-learn`
+ TensorFlow >=1.14.0 or >=2.0.0
+ tf_geometric:
Use one of the following commands below to install tf_geometric (and tensorflow):
```bash
pip install -U tf_geometric # this will not install the tensorflow/tensorflow-gpu package

pip install -U tf_geometric[tf1-cpu] # this will install TensorFlow 1.x CPU version

pip install -U tf_geometric[tf1-gpu] # this will install TensorFlow 1.x GPU version

pip install -U tf_geometric[tf2-cpu] # this will install TensorFlow 2.x CPU version

pip install -U tf_geometric[tf2-gpu] # this will install TensorFlow 2.x GPU version
```


