"""
Contains some detectors specifically designed for the Fruits dataset
"""
import torch
from torch import Tensor
from pyswip import Prolog
from pytorch_ood.api import Detector
from pytorch_ood.utils import extract_features
from pytorch_ood.detector import TemperatureScaling
from torch import nn

is_fruit_to_name = ("false", "true")
color_to_name = ("red", "yellow", "green", "brown", "black", "orange")


label_to_name = (
    "orange",
    "strawberry",
    "blueberry",
    "pepper_green",
    "grape_blue",
    "lemon",
    "apple_granny_smith",
    "papaya",
    "tomato",
    "apple_braeburn",
    "cactus_fruit",
    "peach",
    "apricot",
    "watermelon",
    "pineapple",
    "banana",
    "cantaloupe",
    "cucumber_ripe",
    "cherry",
    "corn",
    "clementine",
    "mango",
    "plum",
    "limes",
    "potato_red",
    "avocado",
    "pear",
    "passion_fruit",
    "pomegranate",
    "onion_white",
    "pepper_red",
    "kiwi",
    "raspberry",
)


class EnsembleDetector(Detector):
    def __init__(self, label_net, color_net):
        self.label_net = label_net
        self.color_net = color_net

    @torch.no_grad()
    def predict(self, x):
        label_conf, label = self.label_net(x).softmax(dim=1).cpu().max(dim=1)
        color_conf, color = self.color_net(x).softmax(dim=1).cpu().max(dim=1)

        score = label_conf + color_conf
        score /= 2.0

        return -score

    def predict_features(self):
        pass

    def fit(self):
        pass

    def fit_features(self):
        pass


class Prologic(Detector):
    """
    Logic
    """

    def __init__(
        self,
        kb: str,
        label_net: nn.Module,
        color_net: nn.Module,
        fruit_net: nn.Module = None,
    ):
        self.kb = Prolog()
        self.kb.consult(kb)
        self.label_net = label_net
        self.color_net = color_net
        self.fruit_net = fruit_net

    def fit(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        results = []

        labels_conf, labels = self.label_net(x).softmax(dim=1).cpu().max(dim=1)
        colors_conf, colors = self.color_net(x).softmax(dim=1).cpu().max(dim=1)

        if self.fruit_net:
            _, signs = self.fruit_net(x).softmax(dim=1).cpu().max(dim=1)
        else:
            signs = torch.ones(size=(x.shape[0],)).bool()

        for label, color, sign in zip(labels, colors, signs):
            nlabel = label_to_name[label.item()]
            ncolor = color_to_name[color.item()]
            is_sign = is_fruit_to_name[sign]

            query = f"is_sat({nlabel}, {ncolor}, {is_sign})"

            response = list(self.kb.query(query))
            if bool(response):
                r = 1.0
            else:
                r = 0.0

            results.append(r)

        valid = torch.tensor(results).cpu()

        return -valid.float()

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass


class PrologOOD(Detector):
    """
    Logic OOD with actual knowledge base
    """

    def __init__(
        self,
        kb: str,
        label_net: nn.Module,
        color_net: nn.Module,
        fruit_net: nn.Module = None,
    ):
        self.kb = Prolog()
        self.kb.consult(kb)
        self.label_net = label_net
        self.color_net = color_net
        self.fruit_net = fruit_net

    def fit(self, loader1, loader2, device):
        pass

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        results = []

        labels_conf, labels = self.label_net(x).softmax(dim=1).cpu().max(dim=1)
        colors_conf, colors = self.color_net(x).softmax(dim=1).cpu().max(dim=1)

        if self.fruit_net:
            fruit_conf, fruit = self.fruit_net(x).softmax(dim=1).cpu().max(dim=1)
        else:
            fruit = torch.ones(size=(x.shape[0],)).bool()
            fruit_conf = torch.ones(size=(x.shape[0],)).float()

        for label, color, isfruit in zip(labels, colors, fruit):
            nlabel = label_to_name[label.item()]
            ncolor = color_to_name[color.item()]
            is_fruit = is_fruit_to_name[isfruit]

            query = f"is_sat({nlabel}, {ncolor}, {is_fruit})"

            response = list(self.kb.query(query))
            if bool(response):
                r = 1.0
            else:
                r = 0.0

            results.append(r)

        valid = torch.tensor(results).cpu()
        scores = (
            # TODO: we included signs conf
            torch.stack([labels_conf, colors_conf, fruit_conf], dim=1)
            .mean(dim=1)
            .cpu()
        )

        return -valid * scores

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass


class PrologOODT(Detector):
    """
    Logic OOD with actual knowledge base
    """

    def __init__(
        self,
        kb: str,
        label_net: nn.Module,
        color_net: nn.Module,
        fruit_net: nn.Module = None,
    ):
        self.kb = Prolog()
        self.kb.consult(kb)
        self.label_net = label_net
        self.color_net = color_net
        self.fruit_net = fruit_net
        self.t = 1

        self.scorer_label = TemperatureScaling(self.label_net)
        self.scorer_color = TemperatureScaling(self.color_net)

    def fit(self, loader_label, loader_color, device):
        print("Fitting with temperature scaling")
        logits_label, y1 = extract_features(loader_label, self.label_net, device)
        logits_color, y2 = extract_features(loader_color, self.color_net, device)

        self.scorer_label.fit_features(logits_label, labels=y1)
        self.scorer_color.fit_features(logits_color, labels=y2)
        print(f"{self.scorer_label.t=}")
        print(f"{self.scorer_color.t=}")

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        results = []

        labels_conf, labels = self.label_net(x).div(self.t).softmax(dim=1).cpu().max(dim=1)
        colors_conf, colors = self.color_net(x).div(self.t).softmax(dim=1).cpu().max(dim=1)

        labels_conf = self.scorer_label(x).cpu()
        colors_conf = self.scorer_color(x).cpu()

        if self.fruit_net:
            fruit_conf, fruit = self.fruit_net(x).div(self.t).softmax(dim=1).cpu().max(dim=1)
        else:
            fruit = torch.ones(size=(x.shape[0],)).bool()
            # fruit_conf = torch.ones(size=(x.shape[0],)).float()

        for label, color, isfruit in zip(labels, colors, fruit):
            nlabel = label_to_name[label.item()]
            ncolor = color_to_name[color.item()]
            is_fruit = is_fruit_to_name[isfruit]

            query = f"is_sat({nlabel}, {ncolor}, {is_fruit})"

            response = list(self.kb.query(query))
            if bool(response):
                r = 1.0
            else:
                r = 0.0

            results.append(r)

        valid = torch.tensor(results).cpu()
        scores = (
            # TODO: we included signs conf
            torch.stack([labels_conf, colors_conf], dim=1)
            .mean(dim=1)
            .cpu()
        )
        # no negative
        return valid * scores

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass
