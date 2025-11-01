#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
yes | conda env remove --name pace-predictor-env

conda env create --file envs/environment.yml
conda activate pace-predictor-env
python -m ipykernel install --user --name your-env-name --display-name "Python (pace-predictor-env)"
