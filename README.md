# Opti-Acoustic Semantic Mapping 

## Requirements 
DCSAM
sonar_oculus 
dino-vit-features (see below) 



## segmentation code based on: dino-vit-features
[[paper](https://arxiv.org/abs/2112.05814)] [[project page](https://dino-vit-features.github.io)]

## Citation
```
@article{amir2021deep,
    author    = {Shir Amir and Yossi Gandelsman and Shai Bagon and Tali Dekel},
    title     = {Deep ViT Features as Dense Visual Descriptors},
    journal   = {arXiv preprint arXiv:2112.05814},
    year      = {2021}
}
```

## Setup
Their code is developed in `pytorch` on and requires the following modules: `tqdm, faiss, timm, matplotlib, pydensecrf, opencv, scikit-learn`.
They use `python=3.9` but the code should be runnable on any version above `3.6`.
They recommend running their code with any CUDA supported GPU for faster performance.
Setup the running environment via Anaconda by running the following commands:
```
$ conda env create -f env/dino-vit-feats-env.yml
$ conda activate dino-vit-feats-env
```
Otherwise, run the following commands in your conda environment:
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch
$ conda install tqdm
$ conda install -c conda-forge faiss
$ conda install -c conda-forge timm 
$ conda install matplotlib
$ pip install opencv-python
$ pip install git+https://github.com/lucasb-eyer/pydensecrf.git
$ conda install -c anaconda scikit-learn
```

## Misc
To run with maxmixture repo, run before running the main roslaunch command
$ python tf_to_odom_node.py
