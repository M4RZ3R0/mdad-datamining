from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from car import Car
from constants import NUM_CLUSTERS, uri, user, password, colors
from neo4j_driver import Neo4JDriver
import pandas as pd

def extract_data():
    # Obtener los coches de Neo4j
    result = driver.get_cars()

    # Crear una lista de objetos Car
    cars = [Car.from_neo4j(record[0], record[1], record[2], record[3]) for record in result]
    
    # Crear un DataFrame de pandas con los datos de los coches
    df = pd.DataFrame([car.to_dict() for car in cars])

    # Devolver el DataFrame
    return df

def k_means(cars):
    # Seleccionar las características numéricas de los datos
    X = cars[["city_lkm", "highway_lkm"]]

    # Crear un modelo K-Means con k=4
    kmeans = KMeans(n_clusters=NUM_CLUSTERS)

    # Ajustar el modelo a los datos
    kmeans.fit(X)

    # Obtener las etiquetas de clúster asignadas a cada coche
    cars["cluster"] = kmeans.labels_

    return (kmeans, cars)

def show_clusters(kmeans, cars):
    # Crear una figura y un conjunto de ejes para la visualización
    _, ax = plt.subplots()

    # Iterar sobre los clústeres y visualizar los datos correspondientes a cada uno
    for i in range(NUM_CLUSTERS):
        cluster = cars[cars["cluster"] == i]
        ax.scatter(cluster["city_lkm"], cluster["highway_lkm"], c=colors[i], label=f"Cluster {i+1}")

    # Agregar los centroides de cada clúster a la visualización
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1], c="k", marker="x", label="Centroids")

    # Agregar leyenda y títulos a la visualización
    ax.legend()
    ax.set_xlabel("Consume en ciudad (L/100km)")
    ax.set_ylabel("Consume en carretera (L/100km)")
    ax.set_title("K-Means Clustering de Coches")

    # Mostrar la visualización
    plt.show()

def predict_fuel_efficiency(cars):
    # Seleccionar características relevantes
    X = cars[["cylinders", "displacement", "transmission", "make", "car_class", "fuel_type"]].copy()
    y = cars["combination_lkm"].copy()

    # Codificar variables categóricas
    encoder = LabelEncoder()
    X["transmission"] = encoder.fit_transform(X["transmission"])
    X["make"] = encoder.fit_transform(X["make"])
    X["car_class"] = encoder.fit_transform(X["car_class"])
    X["fuel_type"] = encoder.fit_transform(X["fuel_type"])

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Entrenar el modelo MLP Regressor
    mlp = MLPRegressor(hidden_layer_sizes=(100), max_iter=50000, random_state=42, alpha=0.04, solver="lbfgs")
    mlp.fit(X_train, y_train)

    # Realizar predicciones sobre el conjunto de prueba
    y_pred_mlp = mlp.predict(X_test)

    # Calcular el error cuadrático medio (Mean Squared Error, MSE)
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    print(f"MSE del modelo MLP: {mse_mlp}")

    # Calcular el coeficiente de determinación (R²)
    r2_mlp = r2_score(y_test, y_pred_mlp)
    print(f"R² del modelo MLP: {r2_mlp}")

    # Visualizar las predicciones utilizando un gráfico de dispersión
    plt.scatter(y_test, y_pred_mlp, alpha=0.5)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.title("MLP Regressor - Predicción de combination_lkm")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "k--", linewidth=2)
    plt.show()

if __name__ == '__main__':
    driver = Neo4JDriver(uri, user, password)
    cars = extract_data()
    #(kmeans, cars) = k_means(cars)
    #show_clusters(kmeans, cars)
    predict_fuel_efficiency(cars)
    driver.close()