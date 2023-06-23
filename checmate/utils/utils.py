import glob, json
import numpy as np
from pathlib import Path
from scipy.stats import norm
from typing import Union, List

from ase.io import read, write
from dpdata.system import MultiSystems

def dp_type_map(fpath:str):
    """
    return: type_map

    Parameters:
        dirpath: where is the file named 'type_map.raw'
    """

    if Path(fpath).is_file():
        with open(fpath, "r") as f:
            type_map = f.read().strip(" \n").split("\n")
    else:
        type_map = None

    return type_map


def get_accuracy(fpath):

    with open(fpath, "r") as f:
        percents = json.load(f).values()

    accuracy = [percent["accuracy"] for percent in percents]
    return accuracy


def get_accuracy_candidate_mean(filespath:str|list):

    accuracy, candidate, mean = [], [], []
    filespath = filespath if isinstance(filespath, list) else glob.glob(filespath)
    for percent_file in sorted(filespath):
        with open(percent_file, "r") as fi:
            txt = json.load(fi)
        accuracy.append(float(txt["accuracy"]))
        candidate.append(float(txt["candidate"]))
        mean.append(float(txt["mean"]))
    return accuracy, candidate, mean


def get_mean_and_sigma(model_devi, column:int=4):

    if isinstance(model_devi, str):
        files = sorted(glob.glob(model_devi))
        assert len(files) > 0

        data = []
        for i in list((np.loadtxt(f)[:, column] for f in files)):
            data.extend(i)

        mean, sigma = norm.fit(data)
    
    else:
        mean, sigma = norm.fit(model_devi)

    return mean, sigma


def get_dir_percent_dict(dirpaths:Union[List[str], List[Path]], percents:list):

    d_p_dict = {}
    for dirpath, p in zip(dirpaths, percents):
        try:
            formula = read(str(Path(dirpath)/"lasp.str"), format="dmol-arc").get_chemical_formula()
            d_p_dict[f"{dirpath} - {formula}"] = p
        except:
            d_p_dict[f"{dirpath}"] = p

    
    return d_p_dict


def f_remove(filepath:str):

    if "*" in filepath:
        for i in glob.glob(filepath):
            Path.unlink(Path(i))
    else:
        Path.unlink(Path(filepath))


def filing(input_files:list[str], output_file):
    """
    put the contents of multiple files into a single file

    Parameters:
        input_files: the collection of files to read
        output_file: the file to write
    """
    if Path(output_file).exists():
        Path.unlink(Path(output_file))
    for file in input_files:
        fi = open(file,"r")
        fo = open(output_file,"a+")
        fo.write(fi.read())
        fi.close()
        fo.close()


def get_dataset_minimum_structs(dirname:str, filename:str, output_dir:str="./"):

    ms = MultiSystems.from_dir(dirname, filename, "deepmd/npy")

    for ls in ms:

        min_e = ls.data['energies'].min()
        min_idx = int(ls.data['energies'].argmin())
        struct = ls[min_idx].to_ase_structure()[0]
        
        assert struct.get_total_energy() == min_e
        write(str(Path(output_dir)/f"min-{struct.get_chemical_formula()}-{min_e/ls.get_natoms():.4f}.cif"), struct, format="cif")   



def modified_abspath_to_relpath(dirname:str) -> Path:

    dirname = Path(dirname).resolve()
    return dirname.relative_to(Path.cwd()) 



