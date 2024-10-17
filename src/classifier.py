import numpy as np


class DigitClassifier:
    def __init__(self, classifier_digit_0, classifier_digit_1, classifier_digit_4, type_model="linear"):
        """
        Inicializa a classe DigitClassifier com os três classificadores.

        Args:
        - classifier0: O classificador para o dígito 0.
        - classifier1: O classificador para o dígito 1.
        - classifier4: O classificador para o dígito 4.
        """
        self.classifier_digit_0 = classifier_digit_0
        self.classifier_digit_1 = classifier_digit_1
        self.classifier_digit_4 = classifier_digit_4
        self.type_model = type_model

    def predict_f0(self, x: np.ndarray) -> np.ndarray:
        """
        Prediz o dígito usando o classificador 0.

        Args:
        - x: Um array numpy com os atributos de entrada.

        Returns:
        - np.ndarray: A predição do classificador para o dígito 0.
        """
        if self.type_model == "PLA":
            return np.array(self.classifier_digit_0.hypothesis(x))
        if self.type_model == "Logistic":
            return np.array(self.classifier_digit_0.predict_classes(x))
        return np.array(self.classifier_digit_0.predict(x))

    def predict_f1(self, x: np.ndarray) -> np.ndarray:
        """
        Prediz o dígito usando o classificador 1.

        Args:
        - x: Um array numpy com os atributos de entrada.

        Returns:
        - np.ndarray: A predição do classificador para o dígito 1.
        """
        if self.type_model == "PLA":
            return np.array(self.classifier_digit_1.hypothesis(x))
        if self.type_model == "Logistic":
            return np.array(self.classifier_digit_1.predict_classes(x))
        return np.array(self.classifier_digit_1.predict(x))

    def predict_f4(self, x: np.ndarray) -> np.ndarray:
        """
        Prediz o dígito usando o classificador 4.

        Args:
        - x: Um array numpy com os atributos de entrada.

        Returns:
        - np.ndarray: A predição do classificador para o dígito 4.
        """
        if self.type_model == "PLA":
            return np.array(self.classifier_digit_4.hypothesis(x))
        if self.type_model == "Logistic":
            return np.array(self.classifier_digit_4.predict_classes(x))
        return np.array(self.classifier_digit_4.predict(x))

    def classify_digit(self, x: np.ndarray) -> int:
        """
        Classifica o dígito com base nas predições dos classificadores.

        Args:
        - x: Um array numpy com os atributos de entrada.

        Returns:
        - int: O dígito classificado (0, 1, 4, ou 5).
        """
        if self.predict_f0(x) == 1:
            return 0
        elif self.predict_f1(x) == 1:
            return 1
        elif self.predict_f4(x) == 1:
            return 4
        else:
            return 5
