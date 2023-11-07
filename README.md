# SPONANA: SPOt bNANA!

## Installation

Prerequisite: Having [Conda](https://docs.conda.io/projects/miniconda/en/latest/) installed in your system. 

```bash
# Clone and enter the repo
git clone --recursive https://github.com/horizon-blue/sponana.git
cd sponana
```

Note: if you already have the repository cloned without the `--recursive` flag, you'll need to clone the 
submodules manually with the following commands:
```bash
git submodule init 
git submodule update
```

Then, you can create a new Conda environment and install all dependencies with a single command:

```bash
# Set up a new conda environment
conda env create -f environment.yml
# A new conda environment called "sponana" should be created.
# To activate the new environment, run the following:
conda activate sponana
```