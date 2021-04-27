# Generating CNNs using a Genetic Algorithm

This code base is designed to generated deep CNN architectures using a Genetic Algorithm. It is based on ideas from the 
resources listed below.

To illustrate its use, this repo is designed to train a neural network on the [MPII dataset](http://human-pose.mpi-inf.mpg.de/).

### Caveats

Before running this, please be aware of the following caveats:

1. This code needs a GPU. It was designed using an Nvidia RTX3070 with 8GB memory, and struggles to design networks with 
   more than 16 layers. There are definitely optimisations that can be made to the code to increase this.
   
1. It takes a very long time to run. Typically, 100 generations can take 2 hours depending on the number of training 
   samples.

1. 



### Installation

The following Python libraries are used:

* Tensorflow 2.4 and supported CUDA/CuDNN version (see Tensorflow documentation)
* OpenCV
* json_tricks
* DEAP
* EasyDict

Also download the MPII dataset and annotations as described here:
https://github.com/microsoft/human-pose-estimation.pytorch#data-preparation

These need to be copied to `./data/mpii/annot/train.json`, `./data/mpii/annot/valid.json`, and `./data/mpii/images/`.

### Running

Run the jupyter notebook at `./notebooks/GA_CNN_Designer.ipynb`.

### Adapting for other scenarios

* Dataloader
* Graphing results (if needed!)
* Training loop
* Population setup, mutations, etc. 


### Improvements / Todo:

1. Look at how to improve memory utilisation in generating neural networks, specifically around padding and 
   concatenate layers.
1. Make the code more generic and configurable for different use cases
1. Add in different layer types, perhaps in a config file for easy adaptation
1. Improve early stopping, add patience > 1!
1. Parameterize and script main loop
1. Make model more likely to pick processing neurons rather than NOP
1. Add in DepthWise Conv2d
1. Replace padding layer with ZeroPadding2D  
1. Migrate notebook to script

### Citations

* Simple Baselines for Human Pose Estimation and Tracking (Microsoft)

MPII loaders, image visualisation, config and logger functionality based on https://github.com/microsoft/human-pose-estimation.pytorch

```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```

* Hands-On Genetic Algorithms with Python (Packt Publishing)

Book available at https://www.packtpub.com/product/hands-on-genetic-algorithms-with-python/9781838557744, code available
at https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python

* Neural Architecture Search with Reinforcement Learning (Barret Zoph, Quoc V. Le)

Paper available at https://arxiv.org/abs/1611.01578v2

```
@misc{zoph2017neural,
      title={Neural Architecture Search with Reinforcement Learning}, 
      author={Barret Zoph and Quoc V. Le},
      year={2017},
      eprint={1611.01578},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```