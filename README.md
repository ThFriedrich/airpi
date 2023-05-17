# AIRPI
AIRPI uses deep learning for the reconstruction of phase images from 4D scanning transmission electron microscopy (4D-STEM) data. 
This repository provides the implementation of the method described in the publication [Phase Object Reconstruction for 4D-STEM using Deep Learning](https://doi.org/10.1093/micmic/ozac002), including the trained convolutinal neural network (CNN). A dockerfile can be found in the '.devcontainer' folder. If you are using vscode, and you have the remote development extension installed, you should be able to open this project in the specified Docker container right away. With this Container you can avoid the manual installation of tensorflow 2.11 and other dependencies. The usage is demonstrated for hdf5-datasets in [this jupyter notebook](run_reconstruction.ipynb)

The code for the training data generation may be found [here](https://github.com/ThFriedrich/ap_data_generation).

## What does AIRPI do?
The process can be divided into two main steps. First, the complex electron wave function is recovered for a convergent beam electron diffraction pattern (CBED) using a convolutional neural network (CNN). Subsequently, a corresponding patch of the phase object is recovered using the phase object approximation. Repeating this for each scan position in a 4D-STEM dataset and combining the patches by complex summation yields the full-phase object. Each patch is recovered from a kernel of 3x3 adjacent CBEDs only, which eliminates common, large memory requirements and enables live processing during an experiment.The CNN can retrieve phase information beyond the aperture angle, enabling super-resolution imaging. The image contrast formation shows a dependence on the thickness and atomic column type. Columns containing light and heavy elements can be imaged simultaneously and are distinguishable. The combination of super-resolution, good noise robustness, and intuitive image contrast characteristics makes the approach unique among live imaging methods in 4D-STEM.
![concept](https://github.com/ThFriedrich/airpi/assets/47680554/1bffe569-ecf1-4a9a-8e86-71e89c2ba9d7)

## Cite
If you use this method in your research, please cite:
```bibtex
@article{Friedrich2023,
    author = {Friedrich, Thomas and Yu, Chu-Ping and Verbeeck, Johan and Van Aert, Sandra},
    title = "{Phase Object Reconstruction for 4D-STEM using Deep Learning}",
    journal = {Microscopy and Microanalysis},
    volume = {29},
    number = {1},
    pages = {395-407},
    year = {2023},
    month = {01},
    issn = {1431-9276},
    doi = {10.1093/micmic/ozac002},
    url = {https://doi.org/10.1093/micmic/ozac002}
}
```
