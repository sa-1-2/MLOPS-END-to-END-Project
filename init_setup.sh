echo [$(date)]: "START"

echo [$(date)]: "Creating env with python 3.8 version"

conda create -p venv python=3.8 -y

echo [$(date)]: "activating the enviornment"

conda activate ./venv

echo [$(date)]: "installing the dev requirement"

pip install -r requirements_dev.txt

echo [$(date)]: "END"
