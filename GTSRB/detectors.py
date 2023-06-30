"""
Contains some detectos specifically designed for the ... dataset
"""

import numpy as np
import torch
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


class LogicOnlyDetector(Detector):
    def __init__(
        self,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        class_to_shape: dict,
        class_to_color: dict,
        sign_net: nn.Module = None,
    ):
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net
        self.class_to_shape = class_to_shape
        self.class_to_color = class_to_color

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass

    def predict(self, x):
        l = self.label_net(x)
        s = self.shape_net(x)
        c = self.color_net(x)

        labels = l.max(dim=1).indices
        shapes_expected = torch.tensor([self.class_to_shape[c.item()] for c in labels])
        colors_expected = torch.tensor([self.class_to_color[c.item()] for c in labels])

        shapes_detected = s.max(dim=1).indices.cpu()
        colors_detected = c.max(dim=1).indices.cpu()

        s1 = l.softmax(dim=1).max(dim=1).values.cpu()
        s2 = s.softmax(dim=1).max(dim=1).values.cpu()
        s3 = c.softmax(dim=1).max(dim=1).values.cpu()

        # shield outlier score
        if self.sign_net:
            o = self.sign_net(x)
            sign_expected = torch.ones(size=(labels.shape[0],))
            sign_detected = o.argmax(dim=1).cpu()
            consistent = (
                (shapes_detected == shapes_expected)
                & (colors_expected == colors_detected)
                & (sign_expected == sign_detected)
            )
        else:
            consistent = (shapes_detected == shapes_expected) & (
                colors_expected == colors_detected
            )

        return (1 - consistent.long()).float()

    def fit(self, *args, **kwargs):
        pass


class SemanticDetector(Detector):
    def __init__(
        self,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        class_to_shape: dict,
        class_to_color: dict,
        sign_net: nn.Module = None,
    ):
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net
        self.class_to_shape = class_to_shape
        self.class_to_color = class_to_color

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass

    def predict(self, x):
        l = self.label_net(x)
        s = self.shape_net(x)
        c = self.color_net(x)

        labels = l.max(dim=1).indices
        shapes_expected = torch.tensor([self.class_to_shape[c.item()] for c in labels])
        colors_expected = torch.tensor([self.class_to_color[c.item()] for c in labels])

        shapes_detected = s.max(dim=1).indices.cpu()
        colors_detected = c.max(dim=1).indices.cpu()

        s1 = l.softmax(dim=1).max(dim=1).values.cpu()
        s2 = s.softmax(dim=1).max(dim=1).values.cpu()
        s3 = c.softmax(dim=1).max(dim=1).values.cpu()

        # shield outlier score
        if self.sign_net:
            o = self.sign_net(x)
            sign_expected = torch.ones(size=(labels.shape[0],))
            sign_detected = o.argmax(dim=1).cpu()
            s4 = o.softmax(dim=1)[:, 1].cpu()  # TODO: assume 1 is shieled class
            v = torch.stack([s1, s2, s3, s4], dim=1).mean(dim=1)
            scores = (
                (shapes_detected == shapes_expected)
                & (colors_expected == colors_detected)
                & (sign_expected == sign_detected)
            )
        else:
            v = torch.stack([s1, s2, s3], dim=1).mean(dim=1)
            scores = (shapes_detected == shapes_expected) & (colors_expected == colors_detected)

        scores = -v * scores.float()
        return scores

    def fit(self, *args, **kwargs):
        pass


class KBDetector(Detector):
    def __init__(
        self,
        kb: str,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        label_file="data/GTSRB/labels.txt",
        sign_net: nn.Module = None,
        debug=False,
    ):
        self.kb = Prolog()
        self.kb.consult(kb)
        self.count = 0
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net
        self.debug = debug

        self.shape_to_name = ("triangle", "circle", "square", "octagon", "inverse_triange")
        self.color_to_name = ("red", "blue", "yellow", "white")

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
            signs = torch.ones(size=(x.shape[0],)).long()
            signs_conf = torch.ones(size=(x.shape[0],)).float()

        for label, shape, color, sign, o in zip(labels, shapes, colors, signs, x):
            # print(label)
            var = f"x{self.count:06d}"  # name for new variable

            nlabel = self.label_to_name[label.item() - 1]
            nshape = self.shape_to_name[shape.item()]
            ncolor = self.color_to_name[color.item()]

            self.kb.assertz(f"has_shape({var}, {nshape})")
            self.kb.assertz(f"has_color({var}, {ncolor})")
            self.kb.assertz(f"has_label({var}, {nlabel})")

            if sign > 0:
                self.kb.assertz(f"sign({var})")

            response = list(self.kb.query(f"consistent({var})", maxresult=1))
            if bool(response):
                r = 1.0
            else:
                r = 0.0

            if self.debug:
                print("-----------------------------")
                print(f"has_shape({var}, {nshape}) [{shape.item()}]")
                print(f"has_color({var}, {ncolor}) [{color.item()}]")
                print(f"has_label({var}, {nlabel}) [{label.item()}]")
                if sign > 0:
                    print(f"sign({var}) [{sign.item()}]")
                else:
                    print(f"not a sign({var}) [{sign.item()}]")

                if r == 1.0:
                    print("consistent")
                else:
                    print("NOT consistent")
                # plt.imshow(np.moveaxis(o.cpu().numpy(), 0, -1))
                # plt.show()
                print("-----------------------------")

            self.count += 1
            results.append(r)

        valid = torch.tensor(results).cpu()
        scores = (
            torch.stack([labels_conf, shapes_conf, colors_conf, signs_conf], dim=1)
            .mean(dim=1)
            .cpu()
        )

        return -valid * scores

    def fit_features(self, x):
        pass

    def predict_features(self, x):
        pass
