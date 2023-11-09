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