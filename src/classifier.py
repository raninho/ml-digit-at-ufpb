import numpy as np
from abc import ABC, abstractmethod

# Classe base abstrata para os classificadores
class BaseClassifier(ABC):
    def __init__(self, classifier):
        self.classifier = classifier

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

# Implementação para o classificador PLA
class PLAClassifier(BaseClassifier):
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.classifier.hypothesis(x))

# Implementação para o classificador Logístico
class LogisticClassifier(BaseClassifier):
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.classifier.predict_classes(x))

# Implementação para o classificador Linear
class LinearClassifier(BaseClassifier):
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.classifier.predict(x))

# Classe para classificar os dígitos
class DigitClassifier:
    def __init__(self, classifier_digit_0, classifier_digit_1, classifier_digit_4, classifier_type="Linear"):
        """
        Inicializa a classe DigitClassifier com os três classificadores.

        Args:
        - classifier_digit_0: O classificador para o dígito 0.
        - classifier_digit_1: O classificador para o dígito 1.
        - classifier_digit_4: O classificador para o dígito 4.
        - classifier_type: Tipo de modelo a ser utilizado ("PLA", "Logistic", ou "Linear").
        """
        self.classifier_digit_0 = self._get_classifier(classifier_digit_0, classifier_type)
        self.classifier_digit_1 = self._get_classifier(classifier_digit_1, classifier_type)
        self.classifier_digit_4 = self._get_classifier(classifier_digit_4, classifier_type)

    def _get_classifier(self, classifier, classifier_type):
        if classifier_type == "PLA":
            return PLAClassifier(classifier)
        elif classifier_type == "Logistic":
            return LogisticClassifier(classifier)
        return LinearClassifier(classifier)

    def classify_digit(self, x: np.ndarray) -> int:
        """
        Classifica o dígito com base nas predições dos classificadores.

        Args:
        - x: Um array numpy com os atributos de entrada.

        Returns:
        - int: O dígito classificado (0, 1, 4, ou 5).
        """
        if self.classifier_digit_0.predict(x) == 1:
            return 0
        elif self.classifier_digit_1.predict(x) == 1:
            return 1
        elif self.classifier_digit_4.predict(x) == 1:
            return 4
        else:
            return 5