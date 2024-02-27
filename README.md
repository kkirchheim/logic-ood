
<div align="center">
<h1>Out-of-Distribution Detection <br> with Logical Reasoning</h1>

<a href="https://openaccess.thecvf.com/content/WACV2024/html/Kirchheim_Out-of-Distribution_Detection_With_Logical_Reasoning_WACV_2024_paper.html">
    <img alt="Template" src="https://img.shields.io/badge/Paper-WACV-0693e3?style=for-the-badge">
</a>

<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white&style=for-the-badge">
</a>
<a href="https://github.com/kkirchheim/pytorch-ood">
    <img alt="Template" src="https://img.shields.io/badge/-PyTorch--OOD-017F2F?style=for-the-badge&logo=github&labelColor=gray">
</a>


<img width=75% src="img/architecture.png" style="display: block;  margin-left: auto; margin-right: auto;"/>

</div>


## Setup


```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U pytorch-ood
conda install pandas seaborn tqdm scikit-learn
```

### Prolog Inference Engine

To run the actual Prolog engine, you will have to install swi-prolog.

**Ubuntu**

```sh
sudo apt install swi-prolog
pip install -U pyswip==0.2.9
```

## Experiments

Results for experiments can be found in the corresponding subdirectories.

## Citation 

If you find this work usefull, please consider citing 
```
@InProceedings{Kirchheim_2024_WACV,
    author    = {Kirchheim, Konstantin and Gonschorek, Tim and Ortmeier, Frank},
    title     = {Out-of-Distribution Detection With Logical Reasoning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2122-2131}
}
```
