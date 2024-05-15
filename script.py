import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

def predict_fuel_type(cars):
    # Seleccionar características relevantes
    X = cars[["year", "city_lkm", "highway_lkm", "cylinders", "displacement", "transmission", "make", "car_class"]].copy()
    y = np.where(cars["fuel_type"] == "gas", 1, 0)

    # Codificar variables categóricas
    encoder = LabelEncoder()
    X["transmission"] = encoder.fit_transform(X["transmission"])
    X["make"] = encoder.fit_transform(X["make"])
    X["car_class"] = encoder.fit_transform(X["car_class"])

    # Reducir el número de instancias de la clase mayoritaria
    num_instances_minority_class = sum(y == 1)  # 1 es la etiqueta para "no gasolina"
    indices_majority_class = np.where(y == 0)[0]
    np.random.shuffle(indices_majority_class)
    indices_majority_class = indices_majority_class[:num_instances_minority_class]
    indices_selected = np.concatenate([np.where(y == 1)[0], indices_majority_class])
    X_balanced = X.iloc[indices_selected]
    y_balanced = y[indices_selected]

    # Dividir el conjunto de datos balanceado en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

    # Entrenar el modelo KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Realizar predicciones sobre el conjunto de prueba
    y_pred = knn.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = knn.score(X_test, y_test)
    print(f"Precisión del modelo: {accuracy}")

    # Calcular el F1-score del modelo
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-score del modelo: {f1}")

    # Etiquetas reales de las clases
    class_names = ['no gas', 'gas']

    # Visualizar las predicciones utilizando una matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot()
    
    plt.show()

if __name__ == '__main__':
    driver = Neo4JDriver(uri, user, password)
    cars = extract_data()
    (kmeans, cars) = k_means(cars)
    show_clusters(kmeans, cars)
    predict_fuel_type(cars)
    driver.close()