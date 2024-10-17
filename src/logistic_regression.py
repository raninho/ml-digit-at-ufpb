import numpy as np
import random
from numpy import linalg as LA
from typing import List, Optional


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 1000, batch_size: int = 20) -> None:
        """
        Inicializa o modelo de Regressão Logística.

        Args:
        - learning_rate: Taxa de aprendizado (default: 0.1).
        - max_iterations: Número máximo de iterações (default: 1000).
        - batch_size: Tamanho do lote (batch) para atualização dos pesos (default: 20).
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.weights: Optional[np.ndarray] = None

    def train(self, training_data: List[List[float]], labels: List[int]) -> None:
        """
        Ajusta o vetor de pesos (weights) usando gradiente descendente com entropia cruzada.

        Args:
        - training_data: Lista de listas contendo os atributos (features) de entrada.
        - labels: Lista contendo os rótulos de saída (labels) correspondentes.
        """
        X = np.array(training_data)
        y = np.array(labels)

        num_samples = X.shape[0]  # Número de exemplos
        num_features = X.shape[1]  # Número de atributos
        weights = np.zeros(num_features, dtype=float)  # Inicializa os pesos com zeros

        for iteration in range(self.max_iterations):
            gradient_sum = np.zeros(num_features, dtype=float)

            # Seleciona um lote aleatório de dados (batch)
            if self.batch_size < num_samples:
                indices = random.sample(range(num_samples), self.batch_size)
                batch_X = [X[i] for i in indices]
                batch_y = [y[i] for i in indices]
            else:
                batch_X = X
                batch_y = y

            # Calcula o gradiente da função de erro para o batch
            for feature_vector, label in zip(batch_X, batch_y):
                gradient_sum += (label * feature_vector) / (1 + np.exp((label * weights).T @ feature_vector))

            gradient = gradient_sum / len(batch_y)

            # Condição de parada: norm do gradiente muito pequeno
            if LA.norm(gradient) < 0.0001:
                break

            # Atualiza os pesos com o gradiente
            weights = weights + (self.learning_rate * gradient)

        self.weights = weights

    def predict_probabilities(self, input_data: np.ndarray) -> np.ndarray:
        """
        Retorna as probabilidades estimadas pela Regressão Logística.

        Args:
        - input_data: Array contendo os atributos de entrada.

        Returns:
        - Array de probabilidades de cada exemplo ser classificado como 1.
        """
        scores = np.dot(input_data, self.weights)
        probabilities = np.exp(scores) / (1 + np.exp(scores))
        return probabilities

    def predict_classes(self, input_data: np.ndarray) -> np.ndarray:
        """
        Realiza a predição binária (classificação) com base nas probabilidades.

        Args:
        - input_data: Array de entrada.

        Returns:
        - Array de rótulos preditos (1 ou -1).
        """
        probabilities = self.predict_probabilities(input_data)
        predicted_labels = np.where(probabilities >= 0.5, 1, -1)
        return predicted_labels

    def get_weights(self) -> Optional[np.ndarray]:
        """
        Retorna o vetor de pesos ajustado pelo modelo.

        Returns:
        - Vetor de pesos (ou None se o modelo não foi ajustado).
        """
        return self.weights

    def calculate_decision_boundary_y(self, x_value: float, intercept_adjustment: float = 0) -> float:
        """
        Calcula o valor de y na fronteira de decisão dado um valor de x para o plano de regressão.

        Args:
        - x_value: Valor de x.
        - intercept_adjustment: Ajuste opcional para o intercepto no valor de y (default: 0).

        Returns:
        - Valor correspondente de y na fronteira de decisão.
        """
        return (-self.weights[0] + intercept_adjustment - self.weights[1] * x_value) / self.weights[2]