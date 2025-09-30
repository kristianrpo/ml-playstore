# ML Play Store — EDA + Baseline

Proyecto de ML con CRISP-DM sobre el dataset de Google Play (Kaggle).
Este repo contiene:
- `notebooks/` (notebooks con las entregas propuestas)
- `report/` (LaTeX + PDF de entregas)
- `pyproject.toml` (dependencias)
- `data/` (base de datos utilizada para el proyecto)

## Requisitos
- Python 3.11+
- [Poetry](https://python-poetry.org/) (`pip install poetry` o instalador oficial)

## 1) Clonar e instalar dependencias
```bash
git clone https://github.com/kristianrpo/ml-playstore.git
cd ml-playstore
poetry install
```
> `poetry install` creará un entorno virtual y resolverá las dependencias del `pyproject.toml`.

## 2) Activar el entorno (Poetry 2.x)
Poetry 2 ya no activa el entorno automáticamente con `poetry shell`.  
Ahora debes correr:
```bash
poetry env activate
```
Esto imprimirá la ruta del script de activación. Ejemplo:
```
source /home/usuario/.cache/pypoetry/virtualenvs/ml-playstore-xxxx-py3.11/bin/activate
```
Copia y pega ese comando en tu terminal para activar el entorno.  
Tu prompt debería mostrar algo como:
```
(ml-playstore-xxxx-py3.11) usuario@PC:~/ml-playstore$
```

## 3) (Una sola vez) Registrar el kernel de Jupyter y el pre-commit
Con el entorno ya activado, registra el kernel:
```bash
python -m ipykernel install --user --name ml-playstore --display-name "Python (ml-playstore)"
```
Además también, definimos el pre-commit para evitar adiciones innecesarias de notebooks:
```bash
  poetry run nbstripout --install
  ```

## 4) Ejecutar Jupyter
Con el entorno activado puedes correr:
```bash
poetry run jupyter lab
# o
poetry run jupyter notebook
```
En Jupyter selecciona el kernel **Python (ml-playstore)** y abre `notebooks/01_eda_baseline.ipynb`.

## 5) Flujo de trabajo usado
- Crear ramas por tarea: `feature/eda-limpieza`, `feature/baseline`, `docs/reporte`.
- Abrir Pull Requests y revisión en pareja.

## 6) Estructura
```
.
├── data/
│   └── google
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── report/
│   ├── entrega1.tex
│   ├── entrega1.pdf
│   └── references.bib
├── src/
├── pyproject.toml
├── README.md
└── .gitignore
```

## 7) Comandos útiles de Poetry
```bash
poetry add <paquete>            # agregar dependencia
poetry add -D <paquete>         # dependencia de desarrollo
poetry update                   # actualizar locks
poetry run <cmd>                # ejecutar un comando dentro del venv
poetry env info                 # info del entorno
poetry env list --full-path     # ver entornos creados
```

## 8) Reproducibilidad
- Usamos `poetry.lock` para fijar versiones.
- Fijar semillas en sklearn: `random_state=42`.
