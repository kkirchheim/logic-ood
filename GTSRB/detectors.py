"""
Contains some detectors specifically designed for the GTSR dataset
"""
import torch
from pytorch_ood.detector import TemperatureScaling
from pytorch_ood.utils import extract_features
from torch import Tensor
from pyswip import Prolog
from pytorch_ood.api import Detector
from torch import nn


class EnsembleDetector(Detector):
    def __init__(self, label_net: nn.Module, shape_net: nn.Module, color_net: nn.Module):
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net

    def predict(self, x):
        l = self.label_net(x)
        s = self.shape_net(x)
        c = self.color_net(x)

        score_l = l.softmax(dim=1).max(dim=1).values
        score_s = s.softmax(dim=1).max(dim=1).values
        score_c = c.softmax(dim=1).max(dim=1).values
        scores = score_l + score_s + score_c
        return -scores

    def fit(self, *args, **kwargs):
        pass

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass


class LogicOOD(Detector):
    def __init__(
        self,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        class_to_shape: dict,
        class_to_color: dict,
        sign_net: nn.Module = None,
        rotation_net: nn.Module = None,
    ):
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net
        self.rotation_net = rotation_net

        self.class_to_shape = class_to_shape
        self.class_to_color = class_to_color

    def fit_features(self, x):
        pass

    def predict_features(self, x: Tensor) -> Tensor:
        raise ValueError

    @torch.no_grad()
    def get_predictions(self, x):

        results = {
            "label": self.label_net(x).cpu(),
            "shape": self.shape_net(x).cpu(),
            "color": self.color_net(x).cpu(),
        }

        if self.sign_net:
            results["sign"] = self.sign_net(x).cpu()

        if self.rotation_net:
            results["rotation"] = self.rotation_net(x).cpu()

        return results

    @torch.no_grad()
    def consistent(self, x, return_predictions=False):
        """
        Determines of the predictions are consistent with the domain knowledge
        """
        p = self.get_predictions(x)

        labels = p["label"].max(dim=1).indices
        shape = torch.tensor([self.class_to_shape[c.item()] for c in labels])
        color = torch.tensor([self.class_to_color[c.item()] for c in labels])

        shape_hat = p["shape"].max(dim=1).indices.cpu()
        color_hat = p["color"].max(dim=1).indices.cpu()

        consistent = (shape_hat == shape) & (color == color_hat)

        if "sign" in p:
            sign_hat = p["sign"]
            consistent = consistent & (sign_hat.argmax(dim=1) == 1)

        if "rotation" in p:
            rotation_hat = p["rotation"]
            consistent = consistent & (rotation_hat.argmax(dim=1) == 0)

        if return_predictions:
            return consistent, p

        return -consistent.float()

    @torch.no_grad()
    def predict(self, x):
        consistent, p = self.consistent(x, return_predictions=True)

        values = []

        for key, value in p.items():
            conf = value.softmax(dim=1).max(dim=1).values.cpu()
            # print(f"{conf.shape=}")
            values.append(conf)

        if "sign" in p:
            values.append(p["sign"].softmax(dim=1).max(dim=1).values.cpu())

        if "rotation" in p:
            values.append(p["rotation"].softmax(dim=1).max(dim=1).values.cpu())

        scores = torch.stack(values, dim=1).mean(dim=1)
        return -scores * consistent.float()

    def fit(self, *args, **kwargs):
        pass


class PrologOOD(Detector):
    """
    Logic OOD with actual knowledge base
    """

    def __init__(
        self,
        kb: str,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        label_file="data/GTSRB/labels.txt",
        sign_net: nn.Module = None,
    ):
        self.kb = Prolog()
        self.kb.consult(kb)
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net

        self.shape_to_name = ("triangle", "circle", "square", "octagon", "inverse_triange")
        self.color_to_name = ("red", "blue", "yellow", "white")
        self.sign_to_name = ("false", "true")

        with open(label_file, "r") as f:
            self.label_to_name = [
                a.strip("'\n").replace(" ", "_").replace("(", "").replace(")", "")
                for a in f.readlines()
            ]

    def fit(self, *args, **kwargs):
        pass

    def predict(self, x) -> torch.tensor:
        results = []

        labels_conf, labels = self.label_net(x).softmax(dim=1).cpu().max(dim=1)
        shapes_conf, shapes = self.shape_net(x).softmax(dim=1).cpu().max(dim=1)
        colors_conf, colors = self.color_net(x).softmax(dim=1).cpu().max(dim=1)

        if self.sign_net:
            signs_conf, signs = self.sign_net(x).softmax(dim=1).cpu().max(dim=1)
        else:
            signs = torch.ones(size=(x.shape[0],)).bool()
            signs_conf = torch.ones(size=(x.shape[0],)).float()

        for label, shape, color, sign in zip(labels, shapes, colors, signs):
            nlabel = self.label_to_name[label.item() - 1]
            nshape = self.shape_to_name[shape.item()]
            ncolor = self.color_to_name[color.item()]
            is_sign = self.sign_to_name[sign]

            query = f"is_sat({nlabel}, {ncolor}, {nshape}, 0, {is_sign})"

            response = list(self.kb.query(query))  # , maxresult=1
            if bool(response):
                r = 1.0
            else:
                r = 0.0

            results.append(r)

        valid = torch.tensor(results).cpu()
        scores = (
            # TODO: we included signs conf
            torch.stack([labels_conf, shapes_conf, colors_conf, signs_conf], dim=1)
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
    Logic OOD with actual knowledge base and Temperature scaling
    """

    def __init__(
        self,
        kb: str,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        label_file="data/GTSRB/labels.txt",
        sign_net: nn.Module = None,
    ):
        self.kb = Prolog()
        self.kb.consult(kb)
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net

        self.shape_to_name = ("triangle", "circle", "square", "octagon", "inverse_triange")
        self.color_to_name = ("red", "blue", "yellow", "white")
        self.sign_to_name = ("false", "true")

        with open(label_file, "r") as f:
            self.label_to_name = [
                a.strip("'\n").replace(" ", "_").replace("(", "").replace(")", "")
                for a in f.readlines()
            ]

        self.scorer_label = TemperatureScaling(self.label_net)
        self.scorer_color = TemperatureScaling(self.color_net)
        self.scorer_shape = TemperatureScaling(self.shape_net)

    def fit(self, loader_label, loader_color, loader_shape, device):
        print("Fitting with temperature scaling")
        logits_label, y1 = extract_features(loader_label, self.label_net, device)
        logits_color, y2 = extract_features(loader_color, self.color_net, device)
        logits_shape, y3 = extract_features(loader_shape, self.shape_net, device)

        print(f"label: {y1.unique()=} {logits_label.shape=}")
        self.scorer_label.fit_features(logits_label, labels=y1)

        print(f"color: {y2.unique()=} {logits_color.shape=}")
        self.scorer_color.fit_features(logits_color, labels=y2)

        print(f"shape: {y3.unique()=} {logits_shape.shape=}")
        self.scorer_shape.fit_features(logits_shape, labels=y3)

        print(f"{self.scorer_label.t=}")
        print(f"{self.scorer_color.t=}")
        print(f"{self.scorer_shape.t=}")

    def predict(self, x) -> torch.tensor:
        results = []

        _, labels = self.label_net(x).softmax(dim=1).cpu().max(dim=1)
        _, shapes = self.shape_net(x).softmax(dim=1).cpu().max(dim=1)
        _, colors = self.color_net(x).softmax(dim=1).cpu().max(dim=1)

        labels_conf = self.scorer_label(x).cpu()
        colors_conf = self.scorer_color(x).cpu()
        shapes_conf = self.scorer_shape(x).cpu()

        if self.sign_net:
            signs_conf, signs = self.sign_net(x).softmax(dim=1).cpu().max(dim=1)
        else:
            signs = torch.ones(size=(x.shape[0],)).bool()
            # signs_conf = torch.ones(size=(x.shape[0],)).float()

        for label, shape, color, sign in zip(labels, shapes, colors, signs):
            nlabel = self.label_to_name[label.item() - 1]
            nshape = self.shape_to_name[shape.item()]
            ncolor = self.color_to_name[color.item()]
            is_sign = self.sign_to_name[sign]

            query = f"is_sat({nlabel}, {ncolor}, {nshape}, 0, {is_sign})"

            response = list(self.kb.query(query))
            if bool(response):
                r = 1.0
            else:
                r = 0.0

            results.append(r)

        valid = torch.tensor(results).cpu()
        scores = (
            # TODO: we included signs conf
            torch.stack([labels_conf, shapes_conf, colors_conf], dim=1)
            .mean(dim=1)
            .cpu()
        )

        # no negative
        return valid * scores

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass
