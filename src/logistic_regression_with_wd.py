import random
from typing import List, Optional

import numpy as np
from numpy import linalg as LA

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score


class LogisticRegressionWithWeightDecay:
    def __init__(self, learning_rate: float = 0.1, max_iterations: int = 1000, batch_size: int = 20, lambda_: float = 0.01) -> None:
        """
        Inicializa o modelo de Regressão Logística com regularização L2 (Weight Decay).

        Args:
        - learning_rate: Taxa de aprendizado (default: 0.1).
        - max_iterations: Número máximo de iterações (default: 1000).
        - batch_size: Tamanho do lote (batch) para atualização dos pesos (default: 20).
        - lambda_: Fator de regularização L2 (default: 0.01).
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.lambda_ = lambda_  # Regularização L2
        self.weights: Optional[np.ndarray] = None

    def train(self, training_data: List[List[float]], labels: List[int]) -> None:
        """
        Ajusta o vetor de pesos (weights) usando gradiente descendente com entropia cruzada e regularização L2 (weight decay).

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

            # Atualiza os pesos com weight decay (regularização L2)
            weights = weights + self.learning_rate * (gradient - self.lambda_ * weights)

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


# Função para ajustar o parâmetro de regularização lambda usando validação cruzada
def tune_regularization_parameter(X, y, lambdas, learning_rate=0.1, max_iterations=1000, batch_size=20):
    """
    Testa diferentes valores de lambda (regularização L2) para o modelo de Regressão Logística.

    Args:
    - X: Conjunto de atributos de entrada (features).
    - y: Conjunto de rótulos de saída (labels).
    - lambdas: Lista de valores de lambda a serem testados.
    - learning_rate: Taxa de aprendizado.
    - max_iterations: Número máximo de iterações.
    - batch_size: Tamanho do batch para o treinamento.

    Returns:
    - Melhor valor de lambda e o modelo treinado correspondente.
    """
    # Dividir os dados em treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_lambda = None
    best_accuracy = 0.0
    best_model = None

    # Percorrer os diferentes valores de lambda
    for lambda_ in lambdas:
        print(f"Treinando com lambda = {lambda_}")

        # Criar e treinar o modelo com o valor de lambda atual
        model = LogisticRegressionWithWeightDecay(learning_rate=learning_rate, max_iterations=max_iterations, batch_size=batch_size,
                                   lambda_=lambda_)
        model.train(X_train, y_train)

        # Fazer predições nos dados de validação
        y_pred = model.predict_classes(X_val)

        # Calcular a acurácia
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Acurácia na validação: {accuracy}")

        # Verificar se essa é a melhor acurácia obtida
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = lambda_
            best_model = model

    print(f"Melhor lambda encontrado: {best_lambda} com acurácia de {best_accuracy}")
    return best_lambda, best_model


def tune_with_kfold(X, y, lambdas, learning_rate=0.1, max_iterations=1000, batch_size=20, k=5):
    """
    Testa diferentes valores de lambda (regularização L2) usando k-fold cross-validation.

    Args:
    - X: Conjunto de atributos de entrada (features).
    - y: Conjunto de rótulos de saída (labels).
    - lambdas: Lista de valores de lambda a serem testados.
    - learning_rate: Taxa de aprendizado.
    - max_iterations: Número máximo de iterações.
    - batch_size: Tamanho do batch para o treinamento.
    - k: Número de folds para a validação cruzada.

    Returns:
    - Melhor valor de lambda.
    - Melhor model
    """
    kf = KFold(n_splits=k)

    best_lambda = None
    best_accuracy = 0.0
    best_model = None

    # Percorrer os diferentes valores de lambda
    for lambda_ in lambdas:
        accuracies = []

        # Cross-validation
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Criar e treinar o modelo
            model = LogisticRegressionWithWeightDecay(learning_rate=learning_rate, max_iterations=max_iterations,
                                       batch_size=batch_size, lambda_=lambda_)
            model.train(X_train, y_train)

            # Predizer nos dados de validação
            y_pred = model.predict_classes(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            accuracies.append(accuracy)

        # Acurácia média para esse lambda
        avg_accuracy = np.mean(accuracies)
        print(f"Lambda: {lambda_}, Acurácia média: {avg_accuracy}")

        # Verificar se é o melhor lambda
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_lambda = lambda_
            best_model = LogisticRegressionWithWeightDecay(learning_rate=learning_rate, max_iterations=max_iterations,
                                       batch_size=batch_size, lambda_=lambda_)

    print(f"Melhor lambda encontrado: {best_lambda} com acurácia média de {best_accuracy}")
    return best_lambda, best_model