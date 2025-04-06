import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Generar datos ficticios de ejemplo
# X: características médicas (edad, presión arterial, colesterol, etc.)
# y: etiquetas (0 = no enfermedad, 1 = enfermedad)
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 pacientes con 5 características cada uno
y = np.random.randint(0, 2, size=(100,))  # Etiquetas binarias (0 o 1)

# Crear el modelo de la red neuronal
model = Sequential([
    Dense(32, input_dim=5, activation='relu'),  # Capa oculta con 32 neuronas
    Dense(16, activation='relu'),              # Capa oculta con 16 neuronas
    Dense(1, activation='sigmoid')             # Capa de salida con activación sigmoide
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=20, batch_size=10)

# Evaluar el modelo
loss, accuracy = model.evaluate(X, y)
print(f'Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}')

# Predicción con nuevos datos
nuevo_paciente = np.array([[0.6, 0.8, 0.7, 0.5, 0.9]])  # Datos de un nuevo paciente
prediccion = model.predict(nuevo_paciente)
print(f'Probabilidad de enfermedad: {prediccion[0][0]:.2f}')
