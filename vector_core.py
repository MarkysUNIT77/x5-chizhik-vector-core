import numpy as np

class VectorInferenceCore:
    def __init__(self, embedding_dim=128):
        # Инициализируем веса в float32 для жесткой экономии памяти ТСД
        self.weights = np.random.randn(embedding_dim, embedding_dim).astype(np.float32)

    def forward(self, input_tokens):
        """
        Векторизованный инференс слоя на чистом C-уровне NumPy
        """
        x = np.array(input_tokens, dtype=np.float32)
        # Быстрое матричное умножение (BLAS инструкции CPU)
        hidden = np.dot(x, self.weights)
        # Правильный векторизованный синус
        return np.sin(hidden)
