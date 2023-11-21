# SPONANA: SPOt bNANA!

## Installation

Prerequisite: Having [Conda](https://docs.conda.io/projects/miniconda/en/latest/) installed in your system. 

```bash
# Clone and enter the repo
git clone https://github.com/horizon-blue/sponana.git
cd sponana
```

Then, you can create a new Conda environment and install all dependencies with a single command:

```bash
# Set up a new conda environment
conda env create -f environment.yml
# A new conda environment called "sponana" should be created.
# To activate the new environment, run the following:
conda activate sponana
```

Alternatively, if you already have a Conda environment and just want to install the Sponana project, you can do so using Pip:

```bash
# install Sponana in editable mode
pip install -e .
```

## To add a model

You can add new models to Sponana project by adding the files to [`src/sponana/models`](src/sponana/models) directory. Then, you can fetch the models from Sponana package index in your notebook. Here's an example of adding the banana model from [`src/sponana/models/banana/banana.sdf`](src/sponana/models/banana/banana.sdf) to the scene:

 ```python
scenario_data = """
directives:
- add_model:
    name: banana
    file: package://sponana/banana/banana.sdf
"""

scenario = load_scenario(data=scenario_data)
# the callback below is necessary to register Sponana models to the package index
station = MakeHardwareStation(
    scenario, meshcat, parser_preload_callback=sponana.utils.configure_parser
)
 ```

## Headless installation
```bash
pip install -e .
pip install pyvirtualdisplay
sudo apt install xvfb
```

## Bayes3D installation
Make sure the python version for the sponana environment is 3.9.

Then, from top level `sponana` dir, in the `sponana` conda environment, run:
```bash
mkdir lib
cd lib
git clone git@github.com:probcomp/bayes3d.git
cd bayes3d
pip install -e .
pip install git+https://github.com/probcomp/genjax.git
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu118
bash download.sh
```
You can then test the installation by running `python demo.py` (from within `lib/bayes3d/`).

All the commands after `cd bayes3d` are instructions from the bayes3d README.md.  There are pointers in that document to how to resolve common errors which arise after Bayes3d installation.