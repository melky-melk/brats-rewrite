# Brats Mri 3D GAN Model (MONAI)

So with this model it is creating a 3d model based on the slices of mri brain the datasets were feeding in. Right now we are basing it heavily of of adams before we move onto doing the ones that arent so closely linked

## Files:
- train_cycling_gen.py is a model that creates new instances of brain scan images

- train_cycling_diff.py is testing the generated models from the python files against real images to see if it is accurate enough. if it is not then the model will keep generating more until it cant tell the differences between them

- utils has all of the helper functions that adam needs

- loops is where the MONAI code is stored, most of it can be found in the MONAI tutorials documentation, that is where the actual information is trained.

- prep.py saves all of the data or downloads it so we can use it. we only need to run this once before we do the actual model otherwise its going to take too long to run. run with `python prep.py`

## Installation

first you gotta install the requirements in the overall isc-demos, then you get the setup from monai https://github.com/Project-MONAI/GenerativeModels/tree/main 

you do this while inside the isc-demos not in brats.

monai also requires its own installations, i got an error that lead me to here
https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies 

so i went inside generative and took their requirements-dev, and colated it in the monai-requirements.txt

download using `pip install -r monai-requirements.txt`
if you are still having issues try 
pip install monai['all']
this should install all the generative files and the readers

using the new isc you need to reinstall CUDA and torch since a new update came out that makes the previous nvidia drivers incompatible with torch 2.1.0

`pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`