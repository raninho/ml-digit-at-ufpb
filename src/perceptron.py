import numpy as np
from typing import List, Optional


class PocketPLA:
    def __init__(self) -> None:
        # Inicializa os pesos como None
        self.weights: Optional[np.ndarray] = None

    def get_weights(self) -> Optional[np.ndarray]:
        # Retorna os pesos atuais
        return self.weights

    def set_weights(self, weights: np.ndarray) -> None:
        # Define os pesos com o valor fornecido
        self.weights = weights

    def execute(self, features: List[List[float]], labels: List[int]) -> None:
        """
        Executa o algoritmo Pocket PLA.

        Args:
        - features: Lista de listas contendo as características (X).
        - labels: Lista contendo os rótulos (y).
        """
        X = np.array(features)
        N = len(X)
        best_error = len(labels)

        # Inicializa os pesos com zeros
        self.weights = np.zeros(len(X[0]))
        best_weights = self.weights.copy()

        # Loop principal para ajustar os pesos
        for _ in range(N):
            # Verifica se o ponto está classificado errado
            for i in range(len(labels)):
                if np.sign(np.dot(self.weights, X[i])) != labels[i]:
                    # Atualiza os pesos
                    self.weights = self.weights + (labels[i] * X[i])
                    # Calcula o erro dentro da amostra (in-sample error)
                    current_error = self.calculate_in_sample_error(X, labels)
                    # Atualiza o melhor erro e os melhores pesos
                    if best_error > current_error:
                        best_error = current_error
                        best_weights = self.weights.copy()

        # Define os melhores pesos ao final
        self.weights = best_weights

    def get_original_y(self, original_x: float) -> float:
        """
        Calcula o valor de y original dado um x.

        Args:
        - original_x: O valor de x.

        Returns:
        - O valor correspondente de y.
        """
        return (-self.weights[0] - self.weights[1] * original_x) / self.weights[2]

    def hypothesis(self, x: np.ndarray) -> int:
        """
        Calcula o valor da hipótese para um ponto x.

        Args:
        - x: O vetor de características.

        Returns:
        - O valor da classificação (1 ou -1).
        """
        return np.sign(np.dot(self.weights, x))

    def calculate_in_sample_error(self, X: np.ndarray, y: List[int]) -> int:
        """
        Calcula o erro dentro da amostra (in-sample error).

        Args:
        - X: O conjunto de características.
        - y: Os rótulos verdadeiros.

        Returns:
        - O número de erros de classificação.
        """
        error: int = 0
        for i in range(len(y)):
            if np.sign(np.dot(self.weights, X[i])) != y[i]:
                error += 1
        return error

def predict_digit(labels: np.ndarray) -> np.ndarray:
    """
    Substitui os valores no array de labels:
    - Se o valor for igual a 1, ele permanece 1
    - Se o valor for diferente de 1, ele é substituído por 5

    :param labels: Array ou lista contendo os valores originais de labels
    :return: Um array NumPy onde os valores são 1 ou 5
    """
    # Substitui 1 por 1 e qualquer outro valor por 5
    transformed_labels: np.ndarray = np.where(labels == 1, 1, 5)

    # Retorna o array transformado
    return transformed_labels

def pocket_pla_prediction(pla: PocketPLA, features: List[np.ndarray]) -> np.ndarray:
    """
    Realiza a predição do dígito usando o Pocket PLA.

    Args:
    - pla: Instância do PocketPLA.
    - features: Lista contendo os dados de entrada para treinamento.

    Returns:
    - Array numpy com as predições dos dígitos.
    """
    y_pla: List[int] = []
    for feature in features:
        # Aplica o Pocket PLA para predizer o valor de y para cada ponto
        y_pla.append(pla.hypothesis(feature))

    # Converte a lista para array numpy e faz a predição final dos dígitos
    y_pred_pla = predict_digit(np.array(y_pla))
    return y_pred_pla
