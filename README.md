

<h1 style="text-align: center">Out-of-Distribution Detection <br>with Logical Reasoning</h1>

<img width=75% src="img/architecture.png" style="display: block;  margin-left: auto; margin-right: auto;"/>


## Setup


```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U pytorch-ood
conda install pandas seaborn tqdm scikit-learn
```

### Prolog Inference Engine (Optional)

To run the actual Prolog engine, you will have to install swi-prolog.

**Ubuntu**

```sh
sudo apt install swi-prolog
pip install -U pyswip==0.2.9
```
