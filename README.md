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

```python
from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="generative",
    packages=find_packages(exclude=[]),
    version="0.2.2",
    description="Installer to help to use the prototypes from MONAI generative models in other projects.",
    install_requires=["monai>=1.2.0rc1"],
)
```

another txt file required for the monai code