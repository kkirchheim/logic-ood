"""

"""
import os
from os.path import join

import click
import torch
from primatenet import PrimateNet, PrimateNetOOD
from pytorch_ood.utils import fix_random_seed
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l, resnet50
from tqdm import tqdm
from utils import test_trans, trans

fix_random_seed(123)
g = torch.Generator()
g.manual_seed(0)


def eval_acc(model, att_index, device, imagenet_root):
    model.eval()
    test_data = PrimateNet(root=imagenet_root, transform=test_trans, train=False)
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=16,
        worker_init_fn=fix_random_seed,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()

        for inputs, y in test_loader:
            labels = y[:, att_index]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the test images: {correct / total:.2%}")


def train_model(att_index, num_classes, epochs, imagenet_root, device, batch_size=32, lr=0.001):
    """
    train a model for the given attribute index
    """
    train_data = PrimateNet(root=imagenet_root, transform=trans, train=True)
    test_data = PrimateNet(root=imagenet_root, transform=test_trans, train=False)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        worker_init_fn=fix_random_seed,
    )

    model = resnet50(num_classes=1000)
    d = torch.load("/home/kirchheim/model_best.pth.tar")["state_dict"]
    newd = {}
    for key, value in d.items():
        newd[key.replace("module.", "")] = value
    model.load_state_dict(newd)

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    # model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # model.fc = model.classifier[1]

    _ = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)  #

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.00001
    )

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        bar = tqdm(train_loader)
        for inputs, y in bar:
            labels = y[:, att_index]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            bar.set_postfix({"loss": running_loss})

        scheduler.step()

        eval_acc(model, att_index, device=device, imagenet_root=imagenet_root)
        model.train()

    return model


def train_oe_model(epochs, device, imagenet_root, batch_size=16, lr=0.001):
    """
    train a model as binary classifier
    """
    train_data = PrimateNet(
        root=imagenet_root, transform=trans, train=True, target_transform=lambda x: 1
    )
    test_data = PrimateNet(
        root=imagenet_root,
        transform=test_trans,
        train=False,
        target_transform=lambda x: 1,
    )
    test_data_ood = PrimateNetOOD(
        root=imagenet_root, train=False, transform=trans, target_transform=lambda x: 0
    )
    train_data_ood = PrimateNetOOD(
        root=imagenet_root, train=True, transform=trans, target_transform=lambda x: 0
    )

    train_loader = DataLoader(
        train_data + train_data_ood,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        worker_init_fn=fix_random_seed,
    )
    test_loader = DataLoader(
        test_data + test_data_ood,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        worker_init_fn=fix_random_seed,
    )

    model = resnet50(num_classes=1000)
    d = torch.load("/home/kirchheim/model_best.pth.tar")["state_dict"]
    newd = {}
    for key, value in d.items():
        newd[key.replace("module.", "")] = value
    model.load_state_dict(newd)

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    # model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # binary
    # model.fc = model.classifier[1]

    _ = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)  #

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.00001
    )

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        bar = tqdm(train_loader)
        for inputs, y in bar:
            labels = y
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            bar.set_postfix({"loss": running_loss})

        scheduler.step()

        correct = 0
        total = 0

        with torch.no_grad():
            model.eval()

            for inputs, y in test_loader:
                labels = y
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the test images: {correct / total:.2%}")

    return model


@click.command()
@click.option("--device", default="cuda:0")
@click.option("--dataset-root", default="../data/")
@click.option("--imagenet-root", default="/data_slow/kirchheim/datasets/imagenet-2012")
@click.option("--n-epochs", default=10)
@click.option("--n-runs", default=10)
@click.option("--lr", default=0.001)
@click.option("--batch-size", default=32)
def main(device, dataset_root, imagenet_root, n_epochs, n_runs, lr, batch_size):
    """
    Train models
    """
    for seed in range(n_runs):
        oe_model = train_oe_model(
            epochs=n_epochs,
            device=device,
            imagenet_root=imagenet_root,
            lr=lr,
            batch_size=batch_size,
        )
        root = join(dataset_root, "models", f"{seed}")
        os.makedirs(root, exist_ok=True)
        print(f"Saving to {root}")
        torch.save(oe_model.cpu(), join(root, "model-oe.pt"))

        for i, att in enumerate(PrimateNet.attributes):
            print(f"Training model for {att}")
            if att == "class":
                num_classes = 16
            else:
                num_classes = 2

            model = train_model(
                att_index=i,
                num_classes=num_classes,
                epochs=n_epochs,
                device=device,
                imagenet_root=imagenet_root,
                lr=lr,
            )

            root = join(dataset_root, "models", f"{seed}")
            os.makedirs(root, exist_ok=True)

            print(f"Saving to {root}")
            torch.save(model.cpu(), join(root, f"model-{att}.pt"))


if __name__ == "__main__":
    main()
