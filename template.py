import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')
list_of_files = [

    ".github/workflows/.gitkeep",
    "src/mlops/__init__.py",
    "src/mlops/components/__init__.py",
    "src/mlops/components/data_ingestion.py",
    "src/mlops/components/data_transformation.py",
    "src/mlops/components/model_trainer.py",
    "src/mlops/components/model_evaluation.py",
    "src/mlops/pipeline/__init__.py",
    "src/mlops/pipeline/training_pipeline.py",
    "src/mlops/pipeline/prediction_pipeline.py",
    "src/mlops/utils/__init__.py",
    "src/mlops/utils/utils.py",
    "src/mlops/logger/log.py",
    "src/mlops/exception/exception.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiment/experiments.ipynb"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir!= "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass