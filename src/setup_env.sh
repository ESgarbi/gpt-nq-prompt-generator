#!/bin/bash

python3 -m venv local_env

source local_env/bin/activate
pip install --upgrade pip

pip install -U datasets transformers pandas
pip install openai
pip install apache_beam
pip install -U accelerate
pip install jsonlines
pip install torch
pip install pyarrow
#python3 train.py... debug attach to process




# apply:
# chmod +x setup_env.sh
# ./setup_env.sh


# python314 pyenv install 3.14
# python3 -m pip install --user pipx
# pip install -U datasets transformers pandas
# pip install openai
# pyenv activate 3.14
# pip install apache_beam
# pip install accelerate -U



# python3 -m venv env_local
# source ./env_local/bin/activate

# # Install packages from requirements.txt
# pip install -r requirements.txt