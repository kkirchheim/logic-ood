"""

"""
from os.path import join

import click
import pandas as pd
import torch
from detectors import EnsembleDetector, PrimateNetLogicDetector, PrimateNetLogicOnlyDetector
from pytorch_ood.detector import (
    EnergyBased,
    Entropy,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    ReAct,
)
from pytorch_ood.utils import OODMetrics, fix_random_seed
from utils import ResultCache, load_all_models, patch_models

fix_random_seed(123)
g = torch.Generator()
g.manual_seed(0)


def fit(models, oe_model, device, cache):
    """ """
    for model in models:
        model.eval()

    label_net = models[0]

    detectors = {
        "LogicOOD": PrimateNetLogicDetector(models),
        "LogicOOD+": PrimateNetLogicDetector(models, oe_model=oe_model),
        "Logic": PrimateNetLogicOnlyDetector(models),
        "Logic+": PrimateNetLogicOnlyDetector(models, oe_model=oe_model),
        "Ensemble": EnsembleDetector([MaxSoftmax(model) for model in models]),
        "ReAct": ReAct(lambda x: x, label_net.fc),
        "MSP": MaxSoftmax(label_net),
        "Energy": EnergyBased(label_net),
        "Mahalanobis": Mahalanobis(label_net.myfeatures, eps=0),
        "ViM": ViM(
            label_net.myfeatures,
            w=label_net.fc.weight,
            b=label_net.fc.bias,
            d=64,
        ),
        "Entropy": Entropy(label_net),
        "MaxLogit": MaxLogit(label_net),
    }

    features = cache.train_features
    ys = cache.train_labels

    print(features.shape, type(features))

    print(f"Classes in fitting data: {ys.unique()}")
    for name in ["ViM", "Mahalanobis", "KLMatching"]:
        if name in detectors:
            print(f"Fitting {name}")
            try:
                detectors[name].fit_features(features, ys, device=device)
            except TypeError:
                detectors[name].fit_features(features, ys)

    return detectors


def evaluate(cache, detectors, device):
    """ """
    results = []

    with torch.no_grad():
        for data_name in cache.dataset_all_logits.keys():
            deep_features = cache.dataset_features[data_name].to(device)
            oe_logits = cache.dataset_oe_logits[data_name].to(device)
            # note: we assume are labels are equal, they should be ...
            labels = cache.dataset_labels[data_name][0].to(device)
            all_logits = [l.to(device) for l in cache.dataset_all_logits[data_name]]

            # evaluate
            for detector_name, detector in detectors.items():
                print(f"OOD Detection for {detector_name}/{data_name}")
                if detector_name in ["ReAct"]:
                    scores = detector.predict(deep_features)
                elif detector_name in ["ViM", "Mahalanobis"]:
                    scores = detector.predict_features(deep_features)

                elif detector_name in ["Logic", "LogicOOD", "Ensemble"]:
                    scores = detector.predict_features(all_logits)

                elif detector_name in ["Logic+", "LogicOOD+"]:
                    scores = detector.predict_features(all_logits, oe_features=oe_logits)

                else:
                    scores = detector.predict_features(all_logits[0])

                metrics = OODMetrics()
                metrics.update(scores, labels)
                r = metrics.compute()
                r.update({"Method": detector_name, "Dataset": data_name})
                print(r)
                results.append(r)

    return results


@click.command()
@click.option("--device", default="cuda:0")
@click.option("--dataset-root", default="../data/")
@click.option("--n-runs", default=10)
def main(device, dataset_root, n_runs):
    results = []

    for seed in range(n_runs):
        root = join(dataset_root, "models", str(seed))
        models, oe_model = load_all_models(root)

        for i, model in enumerate(models):
            model.to(device)

        oe_model.to(device)

        patch_models(models, oe_model)

        cache_path = join(root, "cache.pt")
        print(f"Loading models from {cache_path}")
        cache: ResultCache = torch.load(cache_path, map_location="cpu")

        detectors = fit(models=models, oe_model=oe_model, device=device, cache=cache)
        result = evaluate(detectors=detectors, cache=cache, device=device)

        for r in result:
            r.update({"Seed": seed})

        results += result

    result_df = pd.DataFrame(results)
    result_df.to_csv("results.csv")


if __name__ == "__main__":
    main()
