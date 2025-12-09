# `np_loaders` – Cargador de datos de numpy a Xarray

Este módulo proporciona una infraestructura para cargar, procesar y convertir datos oceanográficos almacenados en archivos `.npy` en datasets de [Xarray](https://docs.xarray.dev/en/stable/).

Está diseñado para integrarse con el framework [`weatherbenchX`](https://weatherbench-x.readthedocs.io/en/latest/wbx_quickstart.html), una de evaluación de predicciones climáticas que incorpora diversas métricas.

---

## Estructura general

- Carga datos de predicción o reanálisis desde archivos `.npy`.
- Aplica máscaras binarias a los datos.
- Extrae fechas desde nombres de archivo.
- Convierte los datos en un `xarray.Dataset` con dimensiones temporales (`init_time`, `lead_time`, `valid_time`).
    - `init_time`: comienzo de la ejecución del modelo para la predicción.
    - `lead_time`: tiempo en el cual la predicción está disponible.
    - `valid_time`:  sería la suma de las anteriores.
- Usa coordenadas desde un fichero adicional.
- Sobreescribe la función `load_chunk` de la clase base de WeaterBench-X de `DataLoader`.

---

## 🧩 Clases principales

| Clase | Descripción |
|-------|-------------|
| `NpLoaders` | Clase base para carga de datos desde `.npy`, aplica máscaras y genera datasets Xarray. Hereda de la clase base de WeatherBench-X `DataLoader`. <br><br>**Argumentos:**<br>- `path`: el directorio con los archivos .npy por fecha.<br>- `variables`: el nombre de las variables en los archivos anteriores.<br>- `extra_files`: archivos adicionales como máscara y coordenadas.<br>- `extra_variables`: nombres de las variables adicionales (`coordinates`, `mask`). |
| `PredictionsFromNumpy` | Carga datos de predicción, organizados por `init_time` y `lead_time`. |
| `TargetsFromNumpy` | Carga datos de análisis (targets) con cálculo de `valid_time`. |

---

## Requisitos

- Para instalar WeatherBenchX ejecutar:
```bash
pip install git+https://github.com/google-research/weatherbenchX.git
```
- Estructura esperada de archivos `.npy`, con fechas codificadas en los nombres como `*_YYYYMMDD.npy`. Los archivos de `Target` (test) se espera que tengan registros de 17 días, mientras que los de `Prediction` se espera que contengan predicciones de 15 días.

- Se necesitan archivos separados para coordenadas de forma `(2, lat, lon)` y para una máscara binaria que filtre entre tierra y mar `(1:mar y 0:tierra -> forma esperada (lat, lon))`.

---

## Uso

```python
from np_loaders import TargetsFromNumpy, PredictionsFromNumpy

extra_files = ["coordinates.npy", "mask.npy"]
extra_vars = ["coordinates", "mask"]

targets = TargetsFromNumpy(
    path="data/targets/",
    variables=["t2m", "z500"],
    extra_files=extra_files,
    extra_variables=extra_vars
)

dataset = targets._load_chunk_from_source()
```

Más ejemplos se pueden encontrar en el notebook de prueba en `/notebooks/test_dataloader_implemented.ipynb`, habría que cargar también las predicciones. 

Para el uso de las métricas este es un ejemplo básico:

```python

    from weatherbenchX.metrics import deterministic
    from weatherbenchX.metrics import base as metrics_base
    from weatherbenchX import aggregation

    metrics = {
        'rmse': deterministic.RMSE(),
        'mae': deterministic.MAE(),
    }

    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, predictions_dataset, targets_dataset
    )

    aggregator = aggregation.Aggregator(
        reduce_dims=["lead_time"], skipna=True
    )

    aggregation_state = aggregator.aggregate_statistics(statistics)
```

Más información sobre las métricas y el funcionamiento de WeatherBench-X se puede encontrar en su [documentación](https://weatherbench-x.readthedocs.io/en/latest/index.html).
