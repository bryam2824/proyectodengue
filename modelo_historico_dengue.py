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





# Mostrar los mejores parámetros encontrados
print("Mejores parámetros:", optimizador.max)

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

# Cargar y preprocesar los nuevos datos
try:
    df = pd.read_csv("datos.csv", sep=";", decimal=",")

    # Reemplazar las comas por puntos y convertir a numérico
    df = df.replace(',', '.', regex=True)
    df.to_csv("archivo_modificado.csv", sep=",", index=False)
    df = df.apply(pd.to_numeric, errors='coerce')  
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(float)

    # Leer los datos modificados
    nuevos_datos = pd.read_csv('archivo_modificado.csv')

    if nuevos_datos.empty:
        raise ValueError("El archivo 'archivo_modificado.csv' está vacío.")

except FileNotFoundError:
    print("Error: El archivo 'archivo_modificado.csv' no se encontró.")
    nuevos_datos = None
except pd.errors.EmptyDataError:
    print("Error: El archivo 'archivo_modificado.csv' está vacío.")
    nuevos_datos = None
except Exception as e:
    print(f"Error inesperado: {e}")
    nuevos_datos = None

# Realizar predicción si los datos fueron cargados correctamente
if nuevos_datos is not None:
    nuevos_datos = nuevos_datos.to_numpy()
    nuevos_datos = scaler.transform(nuevos_datos)

    # Hacer predicciones
    proba_dengue = model.predict(nuevos_datos)
    print(f"Probabilidad de dengue: {proba_dengue[0][0]*100:.2f}%")
else:
    print("No se pudo realizar la predicción debido a errores en la carga de datos.")
