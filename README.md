# Tutorial

## Install necessary tools

It is suggested to create a virtual environment via `conda` for your project.  After your conda environment is activated, run the following command to install necessary tools.

```bash
pip install --user poetry
```

## Install project

```bash
cd checmate/
poetry install
```
There may be a connection error during this process, try a few more times.

## Configure pymatgen potcar
Before you run checmate, you should configure the pymatgen with the potcar files in your laptops/hpc/etc..
if you are using the potcar files from vasp 5.4, you should set 
```bash
pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54
```
if you are using the potcar files from vasp 5.2, you should set 
```bash
pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_52
```
Otherwise, skip the above commands.

In the next, one should generate pymatgen potcar files using
```bash
pmg config -p <original_potentials_dir> mypsp
```
a `mypsp` directory will be created in current directory.
move `mypsp` to `<somewhere>`

configure the path, through which the pymatgen can find the potentials, the `mypsp`
```bash
pmg config --add PMG_VASP_PSP_DIR  <somewhere>/mypsp 
```

We refer interested readers to https://pymatgen.org/installation.html for detailed setup.


## Run commands / scripts 

```bash
checmate -s SETTING WORKFLOW
```
where the argument SETTING is the name of the parameter file in the JSON/YAML format, which should be determined by the user beforehand. And the argument WORKFLOW is the name of a specific workflow.

```bash
checmate --help
```

