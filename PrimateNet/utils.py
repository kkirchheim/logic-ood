"""

"""
from functools import partial
from os.path import join

import torch
from preset import ClassificationPresetEval, ClassificationPresetTrain
from primatenet import PrimateNet
from pytorch_ood.utils import ToRGB
from torchvision.transforms import Compose, InterpolationMode


def patch_models(models, oe_model):
    for model in models:
        model.myfeatures = partial(myfeatures, model)

    oe_model.myfeatures = partial(myfeatures, oe_model)


trans = ClassificationPresetTrain(
    crop_size=224,
    interpolation=InterpolationMode.BILINEAR,
    auto_augment_policy=None,
    random_erase_prob=0.0,
    ra_magnitude=9,
    augmix_severity=3,
    backend="PIL",
)


test_trans = Compose(
    [
        ToRGB(),
        ClassificationPresetEval(
            crop_size=224, interpolation=InterpolationMode.BILINEAR, backend="PIL"
        ),
    ]
)


class ResultCache(object):
    """
    For caching results etc.
    """

    def __init__(self):
        self.dataset_all_logits = {}
        self.dataset_features = {}
        self.dataset_oe_logits = {}
        self.dataset_labels = {}
        self.train_features = None
        self.val_features = None
        self.val_labels = None
        self.train_labels = None

    def clear(self):
        self.dataset_all_logits = {}
        self.dataset_features = {}
        self.dataset_oe_logits = {}
        self.dataset_labels = {}
        self.train_features = None
        self.val_labels = None
        self.val_features = None
        self.train_labels = None


def myfeatures(self, x):
    """
    Custom forward method for models
    """
    # See note [TorchScript super()]
    # resnet
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    # efficientnet
    # x = self.features(x)
    #
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    return x


def load_all_models(root: str):
    # load saved model
    print(f"Loading models from {root}")
    models = []
    for i, att in enumerate(PrimateNet.attributes):
        p = join(root, f"model-{att}.pt")
        print(f"Loading model for {att} from {p}")
        model = torch.load(p)
        models.append(model)
        print(f"FC: {model.fc.weight.shape}")

    print("Loading model for oe")
    oe_model = torch.load(join(root, "model-oe.pt"))
    return models, oe_model
