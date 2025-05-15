import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from bayes_opt import BayesianOptimization
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import json
import sys
import numpy as np
  # Importa modelo entrenado y scaler



# Establecer una semilla para reproducibilidad
np.random.seed(42)

# Generar datos de ejemplo
data = np.random.rand(100, 3)  # 100 muestras con 3 características
targets = np.random.rand(100)  # Valores objetivo (continuos)

# Aplicar clustering jerárquico
clustering = AgglomerativeClustering(n_clusters=3)
clusters = clustering.fit_predict(data)

# Agregar los clusters como una nueva característica
data_with_clusters = np.hstack((data, clusters.reshape(-1, 1)))

# Normalizar los datos
scaler = StandardScaler()
data_with_clusters = scaler.fit_transform(data_with_clusters)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data_with_clusters, targets, test_size=0.2, random_state=42)

# Definir la función objetivo para la optimización bayesiana
# Definir la función objetivo
def optimizar_svm(C, gamma):
    modelo = SVR(C=C, gamma=gamma, kernel='rbf')  # Cambiar de SVC a SVR
    score = cross_val_score(modelo, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    return score


# Definir los límites de búsqueda
parametros = {'C': (0.1, 100), 'gamma': (0.001, 1)}

# Aplicar Bayesian Optimization
optimizador = BayesianOptimization(f=optimizar_svm, pbounds=parametros, random_state=42)
optimizador.maximize(init_points=5, n_iter=10)
            # Definir la función de adquisición

mejores_parametros = optimizador.max['params']
mejor_modelo = SVR(C=mejores_parametros['C'], gamma=mejores_parametros['gamma'], kernel='rbf')
mejor_modelo.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("Mejores parámetros:", optimizador.max)
"""
#Esto te permite ver qué combinaciones han sido más efectivas
valores_C = [p['params']['C'] for p in optimizador.res]
valores_gamma = [p['params']['gamma'] for p in optimizador.res]
scores = [p['target'] for p in optimizador.res]

plt.scatter(valores_C, valores_gamma, c=scores, cmap='viridis')
plt.xlabel("C")
plt.ylabel("Gamma")
plt.title("Evolución de la Optimización Bayesiana")
plt.colorbar(label="Score")
plt.show()
"""
#Esto te ayuda a verificar si la Optimización Bayesiana realmente mejora los resultados.

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Mejores parámetros con GridSearchCV:", grid_search.best_params_)

# Crear el modelo de red neuronal
model = Sequential([
    Dense(64, input_dim=4, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=200, batch_size=70, validation_data=(X_test, y_test))

try:
    with open("datos_usuario.json", "r") as f:
        datos_usuario = json.load(f)

    if not datos_usuario or len(datos_usuario) == 0:
        raise ValueError("Datos insuficientes para ejecutar el modelo.")

    # Continuar con la ejecución del modelo...
except Exception as e:
    print(f"⚠ Error al procesar los datos: {e}")

def ejecutar_modelo(nuevos_datos):
    if nuevos_datos is None or len(nuevos_datos) == 0:
        print("⚠ Error: No se han proporcionado datos para el análisis.")
        return None

    from modelo_dengue import model, scaler
    nuevos_datos_scaled = scaler.transform([nuevos_datos])
    proba_dengue = model.predict(nuevos_datos_scaled)
    return proba_dengue[0][0] * 100



if __name__ == "__main__":
    print("Entrenando el modelo...")
    model.fit(X_train, y_train)
    model.save("modelo_dengue.h5")  # Guarda el modelo entrenado


