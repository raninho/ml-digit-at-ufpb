from typing import List
import pandas as pd

# Função para calcular a intensidade dos pixels de uma imagem
def intensity(X: pd.DataFrame) -> List[float]:
    intensities: List[float] = []  # Lista para armazenar as intensidades calculadas
    for i in range(len(X)):  # Iterar sobre cada linha do DataFrame X
        # Soma todos os valores de pixels na linha, excluindo o rótulo
        pixel_sum: float = X.iloc[i].sum() - X['label'][i]  
        # Calcula a intensidade dividindo a soma dos pixels por 255 (faixa de cor do pixel)
        intensities.append(pixel_sum / 255)  
    return intensities  # Retorna a lista de intensidades


# Função para calcular a simetria vertical e horizontal de imagens em um DataFrame
def symmetry(X: pd.DataFrame) -> List[float]:
    # 1. Simetria Vertical
    vertical_symmetry_list: List[float] = []
    for image_index in range(len(X)):  # Iterar sobre cada imagem no DataFrame X
        vertical_sum: float = 0
        for row in range(28):  # Iterar sobre as linhas da imagem (28x28 pixels)
            for col in range(14):  # Iterar apenas até a metade da largura (14 pixels)
                left_pixel_index: int = row * 28 + col  # Índice do pixel à esquerda
                right_pixel_index: int = row * 28 + (27 - col)  # Índice do pixel à direita
                # Calcula a diferença absoluta entre os pixels espelhados verticalmente
                vertical_sum += abs(X.iloc[image_index, left_pixel_index + 1] - X.iloc[image_index, right_pixel_index + 1])
        # Normaliza a soma pela intensidade de cor dos pixels (dividindo por 255)
        vertical_symmetry_list.append(vertical_sum / 255)

    # 2. Simetria Horizontal
    horizontal_symmetry_list: List[float] = []
    for image_index in range(len(X)):  # Iterar sobre cada imagem no DataFrame X
        horizontal_sum: float = 0
        for col in range(28):  # Iterar sobre as colunas da imagem (28x28 pixels)
            for row in range(14):  # Iterar apenas até a metade da altura (14 pixels)
                top_pixel_index: int = row * 28 + col  # Índice do pixel no topo
                bottom_pixel_index: int = (27 - row) * 28 + col  # Índice do pixel na parte inferior
                # Calcula a diferença absoluta entre os pixels espelhados horizontalmente
                horizontal_sum += abs(X.iloc[image_index, top_pixel_index + 1] - X.iloc[image_index, bottom_pixel_index + 1])
        # Normaliza a soma pela intensidade de cor dos pixels (dividindo por 255)
        horizontal_symmetry_list.append(horizontal_sum / 255)

    # 3. Soma das simetrias vertical e horizontal
    total_symmetry_list: List[float] = [vertical + horizontal for vertical, horizontal in zip(vertical_symmetry_list, horizontal_symmetry_list)]

    return total_symmetry_list  # Retorna a lista de simetrias totais para cada imagem