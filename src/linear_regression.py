import numpy as np
from typing import Optional, List


class LinearRegression:
    def __init__(self):
        """
        Inicializa o modelo de regressão linear com os pesos (coeficientes) indefinidos.
        """
        self.weights: Optional[np.ndarray] = None  # Inicializa os pesos como None

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Ajusta o modelo de regressão linear usando o método dos mínimos quadrados.

        :param features: Matriz de características (_X) (tamanho N x D), onde N é o número de exemplos e D o número de características.
        :param target: Vetor de valores alvo (_y) (tamanho N), os valores reais correspondentes às características.
        """
        # Número de exemplos (linhas) no conjunto de dados
        num_examples: int = len(features)

        # Adiciona uma coluna de 1s para o termo de viés (bias/intercepto)
        X: np.ndarray = np.column_stack((np.ones((num_examples, 1)), features))

        # Calcula (X^T * X)
        xtx: np.ndarray = X.T @ X

        # Calcula a inversa de (X^T * X) e multiplica por X^T
        inv_xtx: np.ndarray = np.linalg.inv(xtx) @ X.T

        # Calcula os pesos (coeficientes) como inv(X^T * X) * X^T * y
        self.weights = inv_xtx @ target

    def predict(self, features: np.ndarray) -> List[float]:
        """
        Faz previsões com base no conjunto de características fornecido.

        :param features: Matriz de características (tamanho M x D), onde M é o número de exemplos a serem previstos.
        :return: Lista de previsões feitas pelo modelo.
        """
        # Adiciona uma coluna de 1s para o termo de viés (bias/intercepto)
        X: np.ndarray = np.column_stack((np.ones((len(features), 1)), features))

        # Retorna as previsões: produto escalar entre os pesos ajustados e cada exemplo de características
        return [np.dot(self.weights, feature_row) for feature_row in X]

    def get_weights(self) -> Optional[np.ndarray]:
        """
        Retorna os pesos (coeficientes) ajustados após o treinamento do modelo.

        :return: Array NumPy dos pesos ajustados, ou None se o modelo ainda não tiver sido treinado.
        """
        return self.weights


class LinearRegressionClassifier:
    def __init__(self):
        self.weights: Optional[np.ndarray] = None  # Inicializa os pesos como None

    def train(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Treina o classificador de regressão logística ajustando os pesos usando a regressão linear.

        :param features: Matriz de características (tamanho N x D), onde N é o número de exemplos e D o número de características.
        :param target: Vetor de valores alvo (tamanho N), os valores reais correspondentes às características.
        """
        linear_regression = LinearRegression()  # Instancia o modelo de regressão linear
        linear_regression.fit(features, target)  # Ajusta o modelo
        self.weights = linear_regression.get_weights()  # Armazena os pesos ajustados

    def predict(self, features: np.ndarray) -> List[int]:
        """
        Realiza previsões com base no conjunto de características fornecido.

        :param features: Matriz de características (tamanho N x D), onde N é o número de exemplos a serem previstos.
        :return: Lista de previsões, onde o sinal da soma ponderada dos pesos define a classe.
        """
        num_examples: int = len(features)
        # Adiciona uma coluna de 1s para o termo de viés (bias/intercepto)
        augmented_features: np.ndarray = np.column_stack((np.ones((num_examples, 1)), features))

        # Retorna as previsões: aplica o sinal da soma ponderada dos pesos para cada exemplo
        return [np.sign(np.dot(self.weights, feature_row)) for feature_row in augmented_features]

    def get_decision_boundary(self, x_value: float, shift: float = 0) -> float:
        """
        Calcula o valor de y da fronteira de decisão (decision boundary) dado um valor de x.

        :param x_value: Valor de x para o qual queremos calcular a fronteira de decisão.
        :param shift: Ajuste opcional para deslocar a fronteira de decisão.
        :return: Valor de y correspondente na fronteira de decisão.
        """
        # A equação da fronteira de decisão é: w0 + w1*x + w2*y = 0, isolando y obtemos:
        return (-self.weights[0] + shift - self.weights[1] * x_value) / self.weights[2]
