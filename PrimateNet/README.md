# PrimateNet

For PrimateNet, we "precompute" all valid states an do not use the KB for performance reasons.

## Train Models
First, we train several models. This can be done via the `train.py` script. You will need the ImageNet dataset
for this.

```
train.py
```

## Extract Features
Extract features, logits etc. from models using the `extract.py` script. Apart from the ImageNet, this should
automatically download all required datasets.

```
extract.py
```

## Evaluate Detectors
Finally, we fit detectors to the extracted features and evaluate using the `evaluate.py` script.

```
evaluate.py
```

Results will be stored in `results.csv`.
