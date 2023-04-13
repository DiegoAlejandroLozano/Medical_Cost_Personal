## Predicción de costos de seguro médico 

Librerías utilizadas: `Pandas`, `Numpy`, `Matplotlib`, `Seaborn` y `Scikit-Learn`

Mediante diferentes algoritmos de regresión de Machine Learning se predice el costo del seguro médico de una persona. Para realizar la predicción se tiene en cuenta factores como: edad, sexo, bmi, hijos, si es fumador y la región a la que pertenece dentro de los Estados Unidos. Se utilizan diferentes algoritmos como regresión lineal, máquinas de vectores de soporte, árboles de decisión y muchos más. En la siguiente tabla se muestran los algoritmos utilizados y los puntajes obtenidos; la tabla se orden de mayor a menor, en función del $R^2$ score test.

|           **Modelo**          	| **$R^2$ train** 	| **$R^2$ test** 	| **RMSR train** 	| **RMSR test** 	|
|:-----------------------------:	|:------------:	|:-----------:	|:--------------:	|:-------------:	|
| Regresión   Árbol de Decisión 	| 0,87         	| 0,87        	| 48             	| 50            	|
| Regresión   Ensamble Bagging  	| 0,89         	| 0,87        	| 46             	| 51            	|
| Regresión   Polinomial        	| 0,84         	| 0,85        	| 53             	| 54            	|
| Regresión   Ensamble Voting   	| 0,81         	| 0,83        	| 55             	| 56            	|
| Regresión   Polinomial SVM    	| 0,81         	| 0,82        	| 49             	| 52            	|
| Regresión   Lineal            	| 0,74         	| 0,77        	| 65             	| 65            	|
| Regresión   Lasso             	| 0,74         	| 0,77        	| 65             	| 65            	|
| Regresión   Ensamble Boosting 	| 0,77         	| 0,77        	| 70             	| 74            	|
| Regresión   Ridge             	| 0,73         	| 0,76        	| 65             	| 66            	|
| Regresión   Lineal SVM        	| 0,68         	| 0,68        	| 59             	| 62            	|

El dataset se puede consultar [aquí](https://www.kaggle.com/datasets/mirichoi0218/insurance).

Para correr el repositorio de forma local se debe crear un entorno virtual con el siguiente comando:

    Python3 -m venv nombre_entorno_virtual

Se debe activar el entorno virtual e instalar las librerías requeridas a través del archivo requirements.txt

    pip install -r requirements.txt
