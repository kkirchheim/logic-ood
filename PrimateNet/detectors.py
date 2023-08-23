from typing import List

import torch
from torch.utils.data import DataLoader

# local
from primatenet import PrimateNet
from pytorch_ood.api import Detector
from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.utils import extract_features
from torch import Tensor


class EnsembleDetector(Detector):
    def __init__(self, detectors):
        self.detectors = detectors

    def predict_features(self, features: List[Tensor]):
        n_samples = features[0].shape[0]

        features = [f.cpu() for f in features]

        for f in features:
            assert f.shape[0] == n_samples

        # calc ensemble scores
        scores = torch.zeros(size=(n_samples,))

        for feature, detector in zip(features, self.detectors):
            # using softmax
            scores += detector.predict_features(feature)

        # uniform weight
        scores /= len(self.detectors)
        return scores

    def predict(self, x):
        n_samples = x.shape[0]
        scores = torch.zeros(size=(n_samples,))

        for n, detector in enumerate(self.detectors):
            with torch.no_grad():
                scores += self.detector(x).cpu()

        return scores / len(self.detectors)

    def fit(self, *args, **kwargs):
        pass

    def fit_features(self, *args, **kwargs):
        pass


class PrimateNetLogicOnlyDetector(Detector):
    """
    Detector based on logic only
    """

    def __init__(
        self,
        models,
        oe_model=None,
    ):

        self.atts = PrimateNet.attributes
        self.models = models
        self.oe_net = oe_model

        self.ref_vectors = torch.zeros(size=(16, 7))

        for v in PrimateNet.data.values():
            self.ref_vectors[v[1]] = torch.tensor(v[2:])

    def _get_world_models(self, x: List[Tensor]) -> Tensor:
        """
        Extract world models from predictions
        """
        n_samples = x[0].shape[0]

        # get actual ref vectors
        world_models = torch.zeros(n_samples, 7)
        y_hat = x[0].argmax(dim=1)

        # TODO: this can be implemented much faster
        for i in range(n_samples):
            # skip first features with class labels
            for n, f in enumerate(x[1:]):
                world_models[i, n] = f[i].argmax()

        return y_hat, world_models

    def _check_consistency(self, y_hat: Tensor, world_models: Tensor) -> Tensor:
        """
        Check consistency of world-models with constraints (reference models)
        """
        # correct attributes?
        consistent = []

        for state, clazz in zip(world_models, y_hat):
            if (state == self.ref_vectors[clazz]).all():
                consistent.append(1.0)
            else:
                consistent.append(0.0)

        consistent = torch.tensor(consistent).long()
        return consistent

    def predict_features(self, features: List[Tensor], oe_features=None):
        """
        first element must be class-membership-predictions, this speeds up the search
        through the possible configurations
        """
        n_samples = features[0].shape[0]

        for f in features:
            # print(f"Input Features: {f.shape}")
            assert f.shape[0] == n_samples

        features = [f.cpu() for f in features]

        y_hat, world_models = self._get_world_models(features)
        consistent = self._check_consistency(y_hat, world_models)

        # if working with outlier exposure
        if self.oe_net and oe_features is not None:
            oe_features = oe_features.cpu()
            oe_consistent = oe_features.argmax(dim=1).long()  # assuming positive class is one
            consistent = consistent & oe_consistent

        return (1 - consistent).float()

    def predict(self, x):
        features = []

        for model in zip(self.models):
            with torch.no_grad():
                z = model(x).cpu()
                features.append(z)

        if self.oe_net is not None:
            oe_features = self.oe_net(x)
        else:
            oe_features = None

        return self.predict_features(features, oe_features)

    def fit(self, *args, **kwargs):
        pass

    def fit_features(self, *args, **kwargs):
        pass


class PrimateNetLogicDetector(Detector):
    """
    Detector
    """

    def __init__(
        self,
        models,
        detector_class=MaxSoftmax,
        detector_kwargs=None,
        detector_weights=None,
        oe_model=None,
    ):
        if detector_kwargs is None:
            detector_kwargs = {}

        self.atts = PrimateNet.attributes
        self.models = models
        self.oe_net = oe_model

        # create all detectors
        self.detectors = [detector_class(model=m, **detector_kwargs) for m in self.models]

        self.ref_vectors = torch.zeros(size=(16, 7))

        for v in PrimateNet.data.values():
            self.ref_vectors[v[1]] = torch.tensor(v[2:])

        if detector_weights is None:
            detector_weights = [1 / len(self.models) for m in self.models]

        self.detector_weights = detector_weights

    def _get_detector_score(self, x: List[Tensor]) -> Tensor:
        n_samples = x[0].shape[0]

        # calc ensemble scores
        scores = torch.zeros(size=(n_samples,))

        for feature, detector, weight in zip(x, self.detectors, self.detector_weights):
            print(f"{feature.shape=}")
            scores += weight * detector.predict_features(feature)

        return scores

    def _get_world_models(self, x: List[Tensor]) -> Tensor:
        """
        Extract world models from predictions
        """
        n_samples = x[0].shape[0]

        # get actual ref vectors
        world_models = torch.zeros(n_samples, 7)
        y_hat = x[0].argmax(dim=1)

        # TODO: this can be implemented much faster
        for i in range(n_samples):
            # skip first features with class labels
            for n, f in enumerate(x[1:]):
                world_models[i, n] = f[i].argmax()

        return y_hat, world_models

    def _check_consistency(self, y_hat: Tensor, world_models: Tensor) -> Tensor:
        """
        Check consistency of world-models with constraints (reference models)
        """
        # correct attributes?
        consistent = []

        for state, clazz in zip(world_models, y_hat):
            if (state == self.ref_vectors[clazz]).all():
                consistent.append(1.0)
            else:
                consistent.append(0.0)

        consistent = torch.tensor(consistent).long()
        return consistent

    def predict_features(self, features: List[Tensor], oe_features=None):
        """
        first element must be class-membership-predictions, this speeds up the search
        through the possible configurations
        """
        n_samples = features[0].shape[0]

        for f in features:
            # print(f"Input Features: {f.shape}")
            assert f.shape[0] == n_samples

        features = [f.cpu() for f in features]

        scores = self._get_detector_score(features)
        y_hat, world_models = self._get_world_models(features)
        consistent = self._check_consistency(y_hat, world_models)

        # if working with outlier exposure
        if self.oe_net and oe_features is not None:
            oe_features = oe_features.cpu()
            oe_consistent = oe_features.argmax(dim=1).long()  # assuming positive class is one
            consistent = consistent & oe_consistent

        return scores * consistent

    @torch.no_grad()
    def predict(self, x):
        features = []

        for model in self.models:
            z = model(x).cpu()
            features.append(z)

        if self.oe_net is not None:
            oe_features = self.oe_net(x)
        else:
            oe_features = None

        return self.predict_features(features, oe_features)

    def fit(self, loader: DataLoader, device="cpu"):

        zs = []
        ys = []

        for model in self.models:
            z, y = extract_features(loader, model, device=device)
            zs.append(z)
            ys.append(y)

        for i, (z, y) in enumerate(zip(zs, ys)):
            self.detectors[i].fit_features(z, y)

    def fit_features(self, *args, **kwargs):
        pass
