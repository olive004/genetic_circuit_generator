#!/bin/sh

pip install jax==0.4.29
pip install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install -U chex
pip install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@f903c39872de43e28b56653efda689bb082cb592#egg=bioreaction

if [ -d "src/bioreaction" ]; then
    echo "Directory src/bioreaction exists."
else
    cd src
    git clone https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git
    cd ..
fi
pip install -e src/bioreaction
