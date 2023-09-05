# DiffPhys-CylinderWakeFlow

This is the repository containing the source codes for **Unsteady Cylinder Wakes from Arbitrary Bodies with Differentiable Physics-Assisted Neural Network** by [`Shuvayan Brahmachary`](https://shuvayanb.github.io/about/) and [`Nils Thuerey`](https://ge.in.tum.de/about/n-thuerey/). The preprint is available on [`arXiv`](https://arxiv.org/abs/2308.04296). 

Additional information: [`project page`](https://ge.in.tum.de/publications/unsteady-cylinder-wakes-from-arbitrary-bodies-with-differentiable-physics-assisted-neural-network/)

## Abstract

This work delineates a hybrid predictive framework configured as a coarse-grained surrogate for reconstructing unsteady fluid flows around multiple cylinders of diverse configurations. The presence of cylinders of arbitrary nature causes abrupt changes in the local flow profile while globally exhibiting a wide spectrum of dynamical wakes fluctuating in either a periodic or chaotic manner. Consequently, the focal point of the present study is to establish predictive frameworks that accurately reconstruct the overall fluid velocity flowfield such that the local boundary layer profile, as well as the wake dynamics, are both preserved for long time horizons. The hybrid framework is realized using a base differentiable flow solver combined with a neural network, yielding a differentiable physics-assisted neural network (DPNN). The framework is trained using bodies with arbitrary shapes, and then it is tested and further assessed on out-of-distribution samples. Our results indicate that the neural network acts as a forcing function to correct the local boundary layer profile while also remarkably improving the dissipative nature of the flowfields. It is found that the DPNN framework clearly outperforms the supervised learning approach while respecting the reduced feature space dynamics. The model predictions for arbitrary bodies indicate that the Strouhal number distribution with respect to spacing ratio exhibits similar patterns with existing literature. In addition, our model predictions also enable us to discover similar wake categories for flow past arbitrary bodies. For the chaotic wakes, the present approach predicts the chaotic switch in gap flows up to the mid-time range.

![Image](Resources/compareVel.png)

## Keywords
Differentiable physics, unsteady cylinder wakes, arbitrary flows, spatio-temporal predictions

## Dataset
The entire dataset used for training and testing the DPNN based-model is available for download [`Link here`](). Refer to respective readme.md files for instructions. The following image shows the 100 experiments that were generated at various spacing ratios using FoamExtend [`FoamExtend`](https://openfoamwiki.net/index.php/Installation/Linux/foam-extend-4.1). 

![Image](Resources/GTData_MeanVel.png)

## FoamExtend
The open source immersed boundary-based solver `FoamExtend v4.0` was used to generate the high-resolution (768 x 512) flowfields for flow past arbitrarily shaped objects on a Cartesian mesh. Refer to [`this link`](https://openfoamwiki.net/index.php/Installation/Linux/foam-extend-4.0) for instalaltion instructions. Recommended open source postprocessing tool [`Paraview`](https://www.paraview.org/). Recommended open source tool for meshing [`Gmsh`](https://gmsh.info/). Validation test cases performed using FoamExtend as well as related source codes and computational meshes will be uploaded [`this link`]().

## Differentiable solver Phiflow
In house developed differentiable flow solver Phiflow [Φ<sub>Flow</sub>] is used as the base *source* solver. The machine learning frameworks [TensorFlow](https://www.tensorflow.org/) is utilised as the python backend. 

## Installation
The following packages/libraries with version have been used:


PhiFlow 2.1.3
Tensorflow 2.3.0
Numpy 1.20.3
Scipy 1.9.3
CUDA 10.0

Installation begins with a simple command for Phiflow
```
!pip install phiflow==2.1.3
```



