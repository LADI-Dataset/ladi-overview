# Setting Up Your Environment
This project uses a conda environment to manage dependencies. If you need `conda`, you can get it from [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) or any of the other Anaconda bundles. Once conda is set up, enter the [requirements](./requirements/) directory and run

```bash
$ conda create ladi-v2-env -y --file conda_requirements -c pytorch -c nvidia
$ conda activate ladi-v2-env
$ pip install -r pip_requirements
```

This will create an environment called `ladi-v2-env` and can be activated with `conda activate ladi-v2-env`

# Exact environment

We have also provided an `environment.yml` file to document the exact configuration used to generate our results. This may be useful for debugging package versions. The environment was created on an Intel-based system with Nvidia graphics running Ubuntu 22.04 (Jammy), and may not reproduce if your system configuration differs. To try and install the exact reproduction environment, you can run.

```bash
conda env create -f environment.yml
```

The environment's name, defined in the yaml file, is `ladi-v2-env` and can be activated with `conda activate ladi-v2-env`. 
