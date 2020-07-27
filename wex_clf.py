"""
This module implements the classifiers for IBM Onewex prediction API models.
adapted from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/estimators/classification/scikitlearn.py
"""
import logging
import requests
from typing import List

import numpy as np

from art.estimators.classification.classifier import Classifier


logger = logging.getLogger(__name__)


class OnewexClassifier(Classifier):
    """
    Wrapper class for onewex models which are accessed via a REST API.
    """

    def __init__(
        self,
        prediction_url: str,
        web_session: "requests.sessions.Session",
        target_classes: List[str]
    ) -> None:
        """
        Create a `Classifier` instance from a onewex model.
        :param prediction_url: rest endpoint for creating predictions (end with /analyze)
        :param web_session: session with basic auth params
        :param target_classes: list of target labels, required for accessing predictions
        """
        super(OnewexClassifier, self).__init__()
        self._prediction_url = prediction_url
        self._web_session = web_session
        self._input_shape = (1,) # just one text feature
        self._classes = target_classes
        self._nb_classes = len(self._classes)

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size=32, nb_epochs=5, verbose=False) -> None:
        """
        Fit the classifier on the training set `(x, y)`.
        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        """
        # not supported/required
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.
        :param x: Test set.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        preds = []
        for observation in x:
            res = self._web_session.post(
                self._prediction_url,
                json={"fields": {"text": observation}},
                verify=False
            )
            pred_res = res.json()["metadata"]["classes"]
            preds.append(self.extract_pred_scores(self._classes, pred_res))

        return np.array(preds)

    def save(self, dirname: str) -> None:
        # not possible
        raise NotImplementedError

    @staticmethod
    def extract_pred_scores(target_labels, wex_dict):
        """
        etxracts prediction scores for all labels from the onewex response
        :param target_labels: list of target labels
        :param wex_dict: metadata.classes dict from onewex api response
        :return: 1d np array with prediction probabilities in order of target_labels
        """
        pred_scores = []
        for label in target_labels:
            pred_scores.append(next((x['probability'] for x in wex_dict if x['label']==label), 0.0))
        return np.array(pred_scores)