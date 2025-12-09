# Documentación del Módulo de Modelos de Malla

En este documento, se describen las funcionalidades del módulo `mesh_models` dentro del directorio `src/seacast_tools`, cuyo propósito consiste en ofrecer una nueva manera de crear modelos de mallas según sea necesario.

## Descripción General
El módulo `mesh_models` proporciona una interfaz para crear y gestionar modelos de malla para su posterior integración con el entrenamiento y evaluación del script `train_model.py`, ofreciendo mallas finales en el formato estándar de PyTorch Geometric.

## Contenido del Módulo

El módulo contiene, aparte del directorio `mesh_models`, los siguientes archivos:
- `create_non_uniform_mesh.py`: Script principal para crear las diferentes clases de mallas, que, además, se le ha dado soporte a la clase de malla uniforme para interactuar mejor con esta forma predeterminada de crear las mallas.
Para su uso más versatil, se han añadido diferentes argumentos para crear mallas personalizadas, siendo estos:
    - `--dataset`: Nombre del conjunto de datos del que se cargarán las coordenadas de los puntos de la cuadrícula, así como información propia del dominio neesaria para la creación de las mallas como puede ser la máscara de agua.
    - `--graph`: Tipo de modelo de estructura de grafos referida a los modelos de `neurallam`, especificando además el directorio donde se guardará el grafo generado, que nos servirá para identificar el tipo de modelo creado, estando por defecto `hierarchical`. Este argumento trabaja en conjunto con el argumento `--hierarchical`.
    - `--plot`: Si se desea generar las diferentes gráficas del grafo a generar (`0` para no mostrar, `1` para mostrar), por defecto `0`.
    - `--levels`: Número de niveles jerárquicos a generar en la malla (desde el nivel más fino hacia el más grueso), por defecto `3`.
    - `--hierarchical`: Indica si se debe generar un grafo jerárquico (`1` para sí, `0` para no), por defecto `1`.
    - `--mesh_type`: Tipo de malla a crear, puede ser `uniform`, `bathymetry`, `random` o `fps`, por defecto `uniform`.
    - `--probability_distribution`: Distribución de probabilidad para generar mallas densificadas, puede ser `mixed_sigmoid` siendo esta una fórmula que junta dos sigmoides, una favoreciendo a la costa y otra favereciendo ligeramente al océano profundo de tal forma que se mantenga un equilibrio en la densificación y se siga representando el océano profundo en la malla o `base`, siendo esta una función de densidad donde se toma la inversa de la raíz cuadrada de la batimetría para densificar nodos, siendo esta función más agresiva en su densificación, por defecto `mixed_sigmoid`.
    - `--crossing_edges`: Añadir o no aristas cruzadas en mallas uniformes (`0` para no añadir, `1` para añadir en forma de X), por defecto `0`.
    - `--uniform_resolution_list`: Lista con el número de nodos por nivel para mallas uniformes, por defecto `[81, 27, 9]`, donde el número total de nodos por nivel será correspondiente, tomando como referencia el primer nivel por defecto, a 81*81 menos los nodos que acaben formandose en la tierra.
    - `--n_connections`: Número de conexiones entre niveles (g2m y m2g), por defecto `1`.
    - `--k_neighboors`: Número de vecinos considerados para conexiones entre niveles (arriba y abajo), por defecto `1`, donde, en este caso, de cada nivel superior saldrá una conexión al nodo más cercano del nivel inferior.
    - `--sampler`: Estrategia de muestreo para mallas `fps`, puede ser `fps` (donde no se llevaría a cabo ninguna estrategia de densificación) o `fps_weighted` (que habilitaría el uso de las estrategias de densificación del argumento `--probability_distribution`), por defecto `fps`.
    - `--nodes_amount`: Lista con el número de nodos por nivel para mallas no uniformes, siendo este argumento de gran importancia para establecer comparativas entre mallas manteniendo el número de nodos exactos entre los mismos, donde dichos números son tomados a partir de las resoluciones especificadas en la malla uniforme con la que queramos compararnos, por defecto `[3568, 394, 45]`.

- `move_files.py`: Script para mover los archivos de malla generados a la carpeta esperada por el script de entrenamiento.
Esta funcionalidad nos permite tener copia de las mallas generadas en caso de que algún procedimiento en la creación de las mismas sea excesivamente largo o, por defecto, se quiera tener una copia de seguridad de las mallas generadas.


    - `--graph_type`: Tipo de grafo a mover, puede ser `hierarchical` u otro tipo compatible; determina la estructura del grafo que se moverá (por defecto: `hierarchical`).
    - `--graph`: Tipo de malla asociada al grafo, también se utiliza como nombre de la carpeta dentro del directorio de origen. Las opciones válidas son:
        - `random`: Malla con nodos colocados aleatoriamente.
        - `uniform`: Malla regular con resolución constante.
        - `bathymetry`: Malla basada en datos de batimetría.
        - `fps`: Malla generada mediante Farthest Point Sampling (muestreo por puntos más distantes).

Por otra parte, tenemos el directorio de `mesh_models`, que contiene los siguientes archivos:

- Un directorio `supplementary_masks` donde se guardan máscaras de aquellas zonas que puedan ser de mayor interés para la evaluación de los modelos en zonas específicas del dominio de estudio, que, en caso de no tenerse, se generarán automáticamente a partir de la máscara de agua.

- Un fichero `mesh_utils.py` que contiene funciones genéricas de utilidad, teniendo por el momento una función soporte para `move_files.py` que permite mover los archivos de malla generados a la carpeta esperada por el script de entrenamiento.

- Un fichero `mesh_metrics.py` que contiene diferentes métricas y procedimientos para la evaluación de las mallas generadas, teniendo en ella diferentes funciones para generar gráficas de las mallas así como conteos de nodosy aristas o media de conexiones entre nodos entre otras métricas similares.

- Un fichero `mesh_connector` que contiene diferentes funciones de conexión entre nodos de la malla del mismo nivel o entre mallas de diferentes niveles, permitiendo mayor flexibilidad a la hora de establecer mecanismos de interconexión en caso de necesitarse.

- Una clase abstracta llamada `Non_uniform_mesh.py` que define la estructura básica de una malla no uniforme, con métodos para crear la malla, generar conexiones y guardar el grafo en formato PyTorch Geometric.

- Las implementaciones de las diferentes mallas no uniformes, que heredan de la clase abstracta `Non_uniform_mesh.py`, son:
    - `BathymetryMesh.py`: Malla basada en datos de batimetría.
    - `RandomMesh.py`: Malla con nodos colocados aleatoriamente.
    - `UniformMesh.py`: Malla regular con resolución constante, con soporte para conexiones cruzadas o no, permitiendo recrear la malla uniforme predeterminada de `neurallam`.
    - `FpsMesh.py`: Malla generada mediante Farthest Point Sampling (muestreo por puntos más distantes).

## Ejemplo de Uso

Para crear las mallas, hay que trasladarse al directorio `src/seacast_tools` y ejecutar el script `create_non_uniform_mesh.py` con los argumentos deseados. Por ejemplo, para crear una malla uniforme con 3 niveles jerárquicos, resoluciones de 27,9,3 respectivamente, sin conexiones cruzadas y guardando las gráficas de las mallas, se puede utilizar el siguiente comando:

```python
python create_non_uniform_mesh.py --dataset atlantic --plot 1 --mesh_type uniform --levels 3 --crossing_edges 0 --uniform_resolution_list 27,9,3 --n_connections 1 --k_neighboors 1
```

si, por otra parte, se quiere utilizar una malla generada en base al algoritmo de Farthest Point Sampling (FPS) con densificación basada en la función de densidad de la batimetría, se puede utilizar el siguiente comando y con una cantidad de nodos equivalente a la malla uniforme generada anteriormente:

```python
python create_non_uniform_mesh.py --dataset atlantic --plot 1 --mesh_type fps --sampler fps_weighted --probability_distribution base --levels 3 --n_connections 1 --k_neighboors 1 --nodes_amount 394,45,5
```

Donde cabe destacar que, cuando se densifica, no se garantiza que los nodos generados estén conectados entre sí en el caso de que la cantidad de los nodos especificados sea excesivamente baja.

Para mover los archivos generados a la carpeta esperada por el script de entrenamiento, se puede utilizar el siguiente comando, donde el argumento de `graph` debe indicar el tipo de malla que se quiera mover, siendo este el nombre de la carpeta dentro del directorio de origen:

```python
python move_files.py --graph_type hierarchical --graph uniform
```

