import numpy as np
import json, copy
from pathlib import Path
from os.path import abspath
from string import Template
from random import randrange
from typing import List, Optional

from ase.atoms import Atom
from ase.io import write, lammpsdata
from pydantic import BaseModel
from pymatgen.io.vasp.sets import DictSet
from pymatgen.io.ase import AseAtomsAdaptor

from .sets import load_config, update_dict, iterdict, dict2string


MODULE_DIR = Path(__file__).resolve().parent


# Vasp Input Modules
class VaspRelaxInput(DictSet):
    """
    Implementation of VaspInputSet utilizing parameters in common case.

    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for vasp.
    template_path: str 
        The path of a template config (format: .yaml or .json).
    """  

    template_path = str(MODULE_DIR/"template"/"VaspRelaxSet.yaml")

    def __init__(self, 
                 structure,
                 user_config: Optional[dict]=None,
                 template_path: str=template_path):    

        structure = AseAtomsAdaptor.get_structure(structure)

        if isinstance(user_config, dict) and user_config:
            modified_keys = {
                "incar": "user_incar_settings",
                "kpoints": "user_kpoints_settings",
                "potcar": "user_potcar_settings"
            }

            for i in modified_keys:
                if user_config.get(i, None) is not None:
                    user_config[modified_keys[i]] = user_config.pop(i)

            super().__init__(structure=structure,
                             config_dict=load_config(template_path),
                             **user_config)
        else:
            super().__init__(structure=structure, config_dict=load_config(template_path))



class VaspStaticInput(VaspRelaxInput):
    """
    Create input files for a static calculation.

    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for vasp.
    template_path: str 
        The path of a template config (format: .yaml or .json).
    """

    template_path = str(MODULE_DIR/"template"/"VaspStaticSet.yaml")

    def __init__(self, 
                 structure,
                 user_config: Optional[dict]=None,
                 template_path: str=template_path):

        super().__init__(structure=structure,
                         user_config=user_config,
                         template_path=template_path)
    
    @staticmethod
    def from_pre_calc(pre_calc_dir:str, files_to_transfer:dict={}):

        if "CHGCAR" not in files_to_transfer:
            files_to_transfer["CHGCAR"] = None

        for i in files_to_transfer.keys():
            fpath = sorted(Path(abspath(pre_calc_dir)).glob(f"{str(i)}*"))
            if fpath:
                files_to_transfer[i] = str(fpath[-1])
            else:
                raise KeyError(f"The file according to the key {i} in files_to_transfer is not exist in pre_calc_dir!")
            
        return files_to_transfer




# Lasp Input Module
class LaspInput():
    """
    Write input files for a LASP calculation

    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for lasp.
    template_path: str 
        The path of a template config (format: .yaml or .json).
    """

    tempate_path = str(MODULE_DIR/"template"/"LaspInSet.yaml")

    def __init__(self,
                 structure,
                 user_config: Optional[dict]=None,
                 template_path: str=tempate_path):

        self.structure = structure
        self.user_config = user_config
        self.template_config = load_config(template_path)


    def block_setting(self, block_dict:dict):

        block_txt = ""

        for key in block_dict:
            content = "\n".join(block_dict[key])
            block_txt = "".join([block_txt, f"%block  {key}\n", content, f"\n%endblock  {key}\n"])
        
        return block_txt


    @property
    def laspin(self):
        """
        update template config according to user config
        """

        laspin = copy.deepcopy(self.template_config)

        if self.user_config:
            update_dict(laspin, self.user_config)

        block_dict = None
        for i in laspin.keys():
            if i.lower() == "block":
                block_dict = laspin.pop(i)
        
        if block_dict is None:
            return dict2string(laspin)
        else:
            return "".join([dict2string(laspin), self.block_setting(block_dict)])
    

    def write_input(self,
                    output_dir: str,
                    make_dir_if_not_present: bool=True):
        """
        Write lasp.in and lasp.str
        """

        output_dir = Path(output_dir)
        if make_dir_if_not_present and not output_dir.exists():
            Path.mkdir(output_dir)

        write(str(output_dir/"lasp.str"),
                 self.structure,
                 format="dmol-arc")

        with open(str(output_dir/"lasp.in"), "wt") as f:
            f.write(self.laspin)




# Lammps Input
class GeneralLammpsInput(BaseModel):
    type_map: List[str]
    models: List[str]


class LammpsInput():
    """
    Write input files for lammps

    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for lammps.
    template_path: str 
        The path of a template config (format: .txt).
    """

    template_path = str(MODULE_DIR/"template"/"LammpsSet.txt")

    def __init__(self,
                 structure,
                 user_config: dict,
                 template_path: str=template_path):

        with open(template_path, "r") as f:
            self.template_config = Template(f.read())
        
        self.type_map = user_config["type_map"]

        if template_path == str(MODULE_DIR/"template"/"LammpsSet.txt"):
            assert GeneralLammpsInput(**user_config)

            mass = "\n".join(
                f"mass            {idx}  {Atom(symbol).mass}" \
                for idx, symbol in enumerate(user_config["type_map"], start=1)
            )
            models = "deepmd  " \
                + "  ".join(str(model) for model in user_config["models"])
            
            user_config = {"mass": mass.strip("\n"), "models": models}

        self.structure = structure
        self.user_config = user_config
        

    @property
    def insimple(self):
        """
        update template config according to user config
        """

        insimple = self.template_config.safe_substitute(**(self.user_config))

        return insimple


    def write_input(self,
                    output_dir: str,
                    make_dir_if_not_present: bool=True):
        """
        Writes input files for a LAMMPS run. Input script is constructed
        from a str template with placeholders to be filled by custom
        settings. Data file is either written from a LammpsData
        instance or copied from an existing file if read_data cmd is
        inspected in the input script. Other supporting files are not
        handled at this moment.

        Parameters
        ----------
        output_dir: str
            Directory to save the input files
        make_dir_if_not_present: bool (True)
            Set to True if you want the directory (and the whole path) 
            to be created if it does not exist. 
        """

        output_dir = Path(output_dir)
        if make_dir_if_not_present and not output_dir.exists():
            Path.mkdir(output_dir)

        lammpsdata.write_lammps_data(str(output_dir/"data.simple"), 
                                     self.structure, specorder=self.type_map)

        with open(str(output_dir/"in.simple"), "w") as f:
            f.write(self.insimple)



# Dp Input Module
class DPTrainInput():
    """
    Writes input json file for a DeepMD-kit training task

    Parameters
    ----------
    systems: list[str]
        the dataset for dp train
    type_map: list[str]
        type_map in DeepMD-kit input json file
    user_config: dict 
        A new config of the input files is disigned by user for dp.
    template_path: str 
        The path of a template config (format: .yaml or .json).
    """

    template_path = str(MODULE_DIR/"template"/"DPTrainInput.json")

    def __init__(self,
                 type_map: List[str],
                 systems: List[str],
                 user_config: Optional[dict]=None,
                 template_path: str=template_path):
    
        self.systems = systems
        self.type_map = type_map
        self.user_config = user_config
        self.template_config = load_config(template_path)


    @property
    def dpinput(self):
        """
        - update template config according to user config
        - set type_map and systems
        - set random seeds
        """

        dpinput = copy.deepcopy(self.template_config)
        if self.user_config:
            update_dict(dpinput, self.user_config)
        
        update_d = {
            "model": {
                "type_map": self.type_map,
                "descriptor": {
                    "seed": randrange(int(1e10))
                },
                "fitting_net": {
                    "seed": randrange(int(1e10))
                }
            },
            "training": {
                "seed": randrange(int(1e10)),
                "training_data": {
                    "systems": self.systems
                }
            }
        }

        if "type_embedding" in dpinput.get("model"):
            update_d["model"]["type_embedding"] = {
                "seed": randrange(int(1e10))
            }

        update_dict(dpinput, update_d)
        return dpinput


    def write_input(self,
                    output_dir: str,
                    make_dir_if_not_present: bool=True):

        output_dir = Path(output_dir)
        if make_dir_if_not_present and not output_dir.exists():
            Path.mkdir(output_dir)

        with open(str(output_dir/"input.json"), "wt") as f:
            json.dump(self.dpinput, f, indent=4)



# Run Input Module
class RunInput():
    """
    Writes input files for a dpdispatcher workflow run.

    Parameters
    ----------
    machine: dict 
        A new config of the machine is disigned by user for dpdispatcher.
    resources: dict 
        A new config of the machine is disigned by user for dpdispatcher.
    resources_temp_style: str
        The key for the module for resources. Only support if resources_temp_path changes.
    *_temp_path: str 
        The path of a template * config (format: .yaml or .json).
    """

    machine_temp_path = str(MODULE_DIR/"template"/"MachineSet.yaml")
    resources_temp_path = str(MODULE_DIR/"template"/"ResourcesSet.yaml")

    def __init__(self,
                 machine: Optional[dict]=None,
                 resources: Optional[dict]=None,
                 resources_temp_key: str="general",
                 machine_temp_path: str=machine_temp_path, 
                 resources_temp_path: str=resources_temp_path):

        self.machine = machine
        self.resources = resources
        self.machine_temp_config = load_config(machine_temp_path)

        if resources_temp_path==str(MODULE_DIR/"template"/"ResourcesSet.yaml"):
            self.resources_temp_config = load_config(resources_temp_path)[resources_temp_key]
        
        else:
            self.resources_temp_config = load_config(resources_temp_path) 


    @property
    def machine_and_resources(self):
        """
        Return
        ----------
        machine and resources in the dictionary format.
        """
        
        machine = copy.deepcopy(self.machine_temp_config)
        if self.machine:
            update_dict(machine, self.machine)

        resources = copy.deepcopy(self.resources_temp_config)
        if self.resources:
            update_dict(resources, self.resources)
        
        return machine, resources 


    def write_input(self,
                    output_dir: str,
                    make_dir_if_not_present: bool=True):

        output_dir = Path(output_dir)
        if make_dir_if_not_present and not output_dir.exists():
            Path.mkdir(output_dir)

        machine, resources = self.machine_and_resources
        with open(str(output_dir/"machine.json"), "wt") as f:
            json.dump({"machine":machine, "resouces":resources}, f, indent=2)



# Cp2k Input Modules
class Cp2kInput():
    """
    Generate input files for CP2K calculation
    
    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for cp2k.
    template_path: str 
        The path of a template config (format: .txt).
    """

    def __init__(self,
                 structure,
                 template_path: str,
                 user_config: Optional[dict]=None):
        
        self.structure = structure
        self.user_config = user_config
        self.template_config = load_config(template_path)


    @property
    def cp2kinp(self):

        cp2kinp = copy.deepcopy(self.template_config)
        
        cell = self.structure.get_cell()
        cell_a = np.array2string(cell[0])[1:-1]
        cell_b = np.array2string(cell[1])[1:-1]
        cell_c = np.array2string(cell[2])[1:-1]
        cell_config = {
            "FORCE_EVAL": {
                "SUBSYS": {
                    "CELL": {
                        "A": cell_a,
                        "B": cell_b,
                        "C": cell_c
                    },
                    "TOPOLOGY": {
                        "COORD_FILE_FORMAT": "XYZ",
                        "COORD_FILE_NAME": "./coord.xyz"
                    }
                }
            }
        }
        
        if self.user_config:
            update_dict(cp2kinp, self.user_config)

        update_dict(cp2kinp, cell_config)
        return cp2kinp


    def write_input(self, 
                    output_dir: str, 
                    make_dir_if_not_present: bool=True):
        
        output_dir = Path(output_dir)
        if make_dir_if_not_present and not output_dir.exists():
            Path.mkdir(output_dir)
        
        write(str(output_dir/"coord.xyz"), self.structure)
        
        input_str = "\n".join(iterdict(self.cp2kinp, out_list=["\n"], loop_idx=0))
        with open(str(output_dir/"cp2k.inp"), "w", encoding="utf-8") as f:
            f.write(input_str.strip("\n"))



class Cp2kRelaxInput(Cp2kInput):
    """
    Generate CP2K input files for geometry optimization

    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for cp2k.
    template_path: str 
        The path of a template config (format: .txt).
    """
    template_path = str(MODULE_DIR/"template"/"Cp2kRelaxInput.json")

    def __init__(self,
                 structure,
                 user_config: Optional[dict]=None,
                 template_path: str=template_path):
        
        super().__init__(structure, user_config=user_config, template_path=template_path)



class Cp2kStaticInput(Cp2kInput):
    """
    Generate CP2K input files for single point energy calculation

    Parameters
    ----------
    structure: Atoms (ase).
    user_config: dict 
        A new config of the input files is disigned by user for cp2k.
    template_path: str 
        The path of a template config (format: .txt).
    """
    template_path = str(MODULE_DIR/"template"/"Cp2kStaticInput.json")

    def __init__(self,
                 structure,
                 user_config: Optional[dict]=None,
                 template_path: str=template_path):
        
        super().__init__(structure, user_config=user_config, template_path=template_path)
