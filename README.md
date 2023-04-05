## Detección de phishing

Librerías utilizadas: `Pandas`, `Numpy`, `Matplotlib`, `Seaborn` y `Scikit-Learn`

Mediante el uso machine learning se determina si un correo electrónico podría representar una amenaza de phishing. Antes de aplicar los diferentes algoritmos de clasificación se realiza el proceso de limpieza y procesamiento de los datos dentro del archivo `limpieza_datos.ipynb`. Una vez se realiza la manipulación del [dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset), se procede entrenar los diferentes algoritmos de clasificación. A continuación, se detalla los algoritmos utilizados y sus respectivos puntajes F1 para los datos de entrenamiento y de prueba (todos los algoritmos utilizados se encuentran dentro del directorio `codigo/`).

| Algoritmo             | F1 Score Train (%) | F1 Score test (%) |
|-----------------------|--------------------|-------------------|
| AdaBoost              | 90,67              | 90,96             |
| Gradient   Boosting   | 90,12              | 90,51             |
| SVM (Gauss)           | 89,90              | 90,11             |
| Bosques   aleatorios  | 89,75              | 90,70             |
| VotingClassifier      | 86,51              | 85,93             |
| Regresión   logística | 85,01              | 84,67             |
| Bagging               | 84,54              | 84,61             |
| SVM (lineal)          | 84,37              | 84,61             |
| Árbol de   decisión   | 83,48              | 83,64             |
| Clasificador   SGD    | 81,09              | 78,53             |

Para correr el repositorio de forma local se debe crear un entorno virtual con el siguiente comando:

    Python3 -m venv nombre_entorno_virtual

Se debe activar el entorno virtual e instalar las librerías requeridas a través del archivo requirements.txt

    pip install -r requirements.txt
