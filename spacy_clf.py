"""
This module implements the classifiers for spacy nlc models.
adapted from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/estimators/classification/scikitlearn.py
"""
import logging
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import random

from art.estimators.classification.classifier import Classifier

from spacy.util import minibatch

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)


# pylint: disable=C0103
def SpacyClassifier(
    model: "spacy.language.Language"
) -> "SpacyClassifier":
    """
    Create a `Classifier` instance from a spacy Language model. This is a convenience function that
    instantiates the correct wrapper class for the given spacy model.
    :param model: spacy model.
    """
    if model.__class__.__module__.split(".")[0] != "spacy":
        raise TypeError("Model is not an spacy model. Received '%s'" % model.__class__)

    spacy_name = model.__class__.__name__

    # This basic class at least generically handles `fit`, `predict` and `save`
    return SpacyClassifier(model)


class SpacyClassifier(Classifier):
    """
    Wrapper class for spacy classifier models.
    """

    def __init__(
        self,
        model: "spacy.language.Language",
        target_classes: List[str] = None,
        architecture: str = 'simple_cnn',
        exclusive_classes: bool = True
    ) -> None:
        """
        Create a `Classifier` instance from a spacy classifier model.
        :param model: spacy classifier model.
        :param target_classes: list of target labels, if None, derived from passed model
        :param architecture: spacy model architecture to use
        :param exclusive_classes: if True, only one class label should be true
        """
        super(SpacyClassifier, self).__init__()
        self._model = model
        self._input_shape = (1,) # just one text feature
        self._classes = target_classes
        if target_classes is None:
            self._classes = self._get_classes()
        self._nb_classes = len(self._classes)
        self._architecture = architecture
        self._exclusive_classes = exclusive_classes
        if not 'textcat' in self._model.pipe_names:
            self._init_spacy_model()

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size=32, nb_epochs=5, verbose=False) -> None:
        """
        Fit the classifier on the training set `(x, y)`.
        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        """
        texts = [str(t) for t in x] # convert from np.str to str
        cats = [{'cats': {str(cat): bool(y_c[idx]) for idx, cat in enumerate(self._classes)}} for y_c in y]
        train_data = list(zip(texts, cats))

        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self._model.pipe_names if pipe not in pipe_exceptions]

        with self._model.disable_pipes(*other_pipes): # only train textcat
            optimizer = self._model.begin_training()
            for i in range(nb_epochs):
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_size)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self._model.update(texts, annotations, sgd=optimizer, drop=0.2)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.
        :param x: Test set.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return np.array([np.fromiter(self._model(str(text)).cats.values(), dtype=float) for text in x])

    def save(self, dirname: str) -> None:
        self._model.to_disk(dirname)

    def _get_classes(self) -> List["String"]:
        # fails if textcat not present
        return self._model.get_pipe("textcat").labels

    def _init_spacy_model(self):
        textcat = self._model.create_pipe(
            "textcat", config={"exclusive_classes": self._exclusive_classes, "architecture": self._architecture}
        )
        self._model.add_pipe(textcat, last=True)
        for cat in self._classes:
            textcat.add_label(cat)

