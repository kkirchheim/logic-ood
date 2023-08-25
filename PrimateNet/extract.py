"""

"""
from functools import partial
from os.path import join

import click
import torch
from primatenet import PrimateNet
from pytorch_ood.dataset.img import (
    FoolingImages,
    ImageNetA,
    ImageNetO,
    ImageNetR,
    Textures,
    NINCO,
    iNaturalist,
)
from pytorch_ood.utils import ToUnknown, fix_random_seed
from torch.utils.data import DataLoader
from train import eval_acc
from utils import ResultCache, load_all_models, myfeatures, test_trans

fix_random_seed(123)
g = torch.Generator()
g.manual_seed(0)


def patch_models(models, oe_model):
    for model in models:
        model.myfeatures = partial(myfeatures, model)

    oe_model.myfeatures = partial(myfeatures, oe_model)


@torch.no_grad()
def extract(models, oe_model, device, imagenet_root, dataset_root) -> ResultCache:
    for model in models:
        model.eval()

    cache = ResultCache()

    extract_train(cache, models[0], device, imagenet_root)

    datasets = {
        d.__name__: d
        for d in (FoolingImages, Textures, ImageNetO, ImageNetR, ImageNetA, NINCO, iNaturalist)
    }
    data_in = PrimateNet(
        root=imagenet_root,
        transform=test_trans,
        train=False,
        target_transform=lambda y: int(y[0]),
    )

    for data_name, dataset_c in datasets.items():
        data_out = dataset_c(
            root=dataset_root,
            transform=test_trans,
            target_transform=ToUnknown(),
            download=True,
        )
        loader = DataLoader(
            data_in + data_out,
            batch_size=64,
            shuffle=False,
            worker_init_fn=fix_random_seed,
            num_workers=12,
        )

        print("Extracting logits from all models")
        extract_logits(cache, data_name, device, loader, models)

        print("Extracting OE Logits")
        extract_oe_logits(cache, data_name, device, loader, oe_model)

        print("Extracting deep features")
        extract_deep_features(cache, models[0], data_name, device, loader)

    return cache


def extract_deep_features(cache, model, data_name, device, loader):
    deep_features = []
    for x, y in loader:
        deep_features.append(model.myfeatures(x.to(device)))
    deep_features = torch.cat(deep_features, dim=0).to(device)
    print(f"{deep_features.shape=}")
    cache.dataset_features[data_name] = deep_features


def extract_oe_logits(cache, data_name, device, loader, oe_model):
    logits = []
    for x, y in loader:
        logits.append(oe_model(x.to(device)))
    oe_logits = torch.cat(logits, dim=0).to(device)
    print(f"{len(oe_logits)=}")
    cache.dataset_oe_logits[data_name] = oe_logits


def extract_logits(cache, data_name, device, loader, models):
    labels, all_logits = [], []
    for n, model in enumerate(models):
        print(f"Model {n}")

        logits, ys = [], []
        for x, y in loader:
            logits.append(model(x.to(device)))
            ys.append(y.to(device))
        logits, ys = torch.cat(logits, dim=0).to(device), torch.cat(ys, dim=0).to(device)

        print(f"{logits.shape=}")
        all_logits.append(logits)
        labels.append(ys)
    # update cache
    print(f"{len(labels)=}")
    print(f"{len(all_logits)=}")
    cache.dataset_labels[data_name] = labels
    cache.dataset_all_logits[data_name] = all_logits


def extract_train(cache, model, device, imagenet_root):
    train_data = PrimateNet(
        root=imagenet_root,
        transform=test_trans,
        target_transform=lambda y: int(y[0]),
        train=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=False,
        num_workers=16,
        worker_init_fn=fix_random_seed,
    )

    print("Extracting features for training")
    features, ys = [], []
    for x, y in train_loader:
        features.append(model.myfeatures(x.to(device)))
        ys.append(y.to(device))
    features, ys = torch.cat(features, dim=0).cpu(), torch.cat(ys, dim=0).cpu()

    # update cache
    cache.train_features = features
    cache.train_labels = ys


@click.command()
@click.option("--device", default="cuda:0")
@click.option("--dataset-root", default="../data/")
@click.option("--imagenet-root", default="/data_slow/kirchheim/datasets/imagenet-2012")
@click.option("--n-runs", default=10)
def main(device, dataset_root, imagenet_root, n_runs):
    """
    Extract features etc. from models
    """

    for seed in range(n_runs):
        root = join(dataset_root, "models", f"{seed}")
        models, oe_model = load_all_models(root)

        for i, (name, model) in enumerate(zip(PrimateNet.attributes, models)):
            model.to(device)
            print(f"Evaluating {name}")
            with torch.no_grad():
                eval_acc(model, i, device=device, imagenet_root=imagenet_root)

        oe_model.to(device)

        patch_models(models, oe_model)

        with torch.no_grad():
            cache = extract(
                models=models,
                device=device,
                oe_model=oe_model,
                dataset_root=dataset_root,
                imagenet_root=imagenet_root,
            )

        cache_path = join(dataset_root, "models", f"{seed}", "cache.pt")
        print(f"Saving cache to {cache_path}")
        torch.save(cache, cache_path)

        for model in models:
            model.cpu()

    print("Done")


if __name__ == "__main__":
    main()
