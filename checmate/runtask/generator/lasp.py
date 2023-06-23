import copy
from pathlib import Path
from typing import List, Union

from ase.atoms import Atoms
from pydantic import BaseModel
from dpdispatcher.submission import Task

from . import GeneralUserConfig, BaseTaskGeneration, GeneralFactaryOutput
from .vasp import VaspTaskGeneration
from .cp2k import Cp2kTaskGeneration
from .lammps import LammpsTaskGeneration
from ... import wflog
from ...pretask.inputs import LaspInput



# Lasp Modules
class LaspTaskConfig(BaseModel):
    command: str = "lasp"
    forward_files: List[str] = ["lasp.in", "lasp.str"]
    backward_files: List[str] = ["lasp.err", "lasp.log", "allfor.arc", "allstr.arc", 
                                 "all.arc", "best.arc", "Badstr.arc", "lasp.out", "SSWtraj"]
    errlog: str = "lasp.err"
    outlog: str = "lasp.log"


class PotentialTaskConfig(BaseModel):
    forward_files: List[str] = []
    backward_files: List[str] = []


class LaspTaskGeneration(BaseTaskGeneration):

    def __init__(
        self,
        lasp_config: Union[dict, GeneralUserConfig],
        potential_config: Union[dict, GeneralUserConfig]={},
        whether_to_cover: bool=False       
    ):  
        self.lasp_config = copy.deepcopy(GeneralUserConfig.parse_obj(lasp_config))
        self.potential_config = copy.deepcopy(GeneralUserConfig.parse_obj(potential_config))
        self.whether_to_cover = whether_to_cover

    
    def gen_task_inputs(self, structure:Atoms, task_dir:str, inputclass=LaspInput):

        inputfile = (Path(task_dir)/"lasp.in")
        template_path = self.lasp_config.template_path
        if template_path:
            inputs = inputclass(
                structure=structure,
                user_config=self.lasp_config.params,
                template_path=template_path)
        else:
            inputs = inputclass(
                structure=structure,
                user_config=self.lasp_config.params)

        if not inputfile.exists():
            inputs.write_input(output_dir=task_dir)
        else:
            wflog.info("The input files of lasp task has already existed! Please confirm whether overwrite. By default, it is false.")
            if self.whether_to_cover:
                inputs.write_input(output_dir=task_dir) 

    
    def gen_potential_inputs(self, task_dir:str, structure:Atoms):

        potentials = {
            "vasp": VaspTaskGeneration,
            "cp2k": Cp2kTaskGeneration,
            "lammps": LammpsTaskGeneration
        }

        file_pairs = {
            "vasp": ["INCAR", "KPOINTS", "POTCAR"], 
            "cp2k": ["cp2k.inp", "coord.xyz"], 
            "lammps": ["data.simple", "in.simple"]
        }
        
        potential = self.lasp_config.params["potential"]
        potential_class = potentials.get(str.lower(potential), None)

        if str.lower(potential) == "nn":
            pass

        elif potential_class is None:
            raise KeyError(f"Unsupported potential function method. Supported: {potentials.keys()}.")
        
        else:
            potential_class(
                self.potential_config,
                whether_to_cover=self.whether_to_cover
            ).gen_task_inputs(task_dir=task_dir, structure=structure)

            p_forward_files = self.potential_config.dpdispatcher.task.get("forward_files", [])
            self.potential_config.dpdispatcher.task["forward_files"] = list(set(p_forward_files + file_pairs[str.lower(potential)]))



    def get_task(self, task_dir:str, structure:Atoms):

        self.gen_task_inputs(task_dir=task_dir, structure=structure)
        self.gen_potential_inputs(task_dir=task_dir, structure=structure)

        task_config = LaspTaskConfig.parse_obj(self.lasp_config.dpdispatcher.task)
        potential_task_config = PotentialTaskConfig.parse_obj(self.potential_config.dpdispatcher.task)

        task_config.forward_files += potential_task_config.forward_files
        task_config.backward_files += potential_task_config.backward_files
        
        return Task(
                    task_work_path=Path(task_dir).name,
                    **(task_config.dict())
                )



def lasp_task_factory(
        lasp_configs: Union[List[dict], List[GeneralUserConfig], dict, GeneralUserConfig],
        potential_configs: Union[List[dict], List[GeneralUserConfig], dict, GeneralUserConfig], 
        structures: List[Atoms], 
        output_dir:str=".", 
        whether_to_cover:bool=False):

    lasp_configs = [lasp_configs] if not isinstance(lasp_configs, list) else lasp_configs
    potential_configs = [potential_configs] if not isinstance(potential_configs, list) else potential_configs
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    task_dirs, task_list = [], []
    assert len(lasp_configs) != 0 and len(structures) != 0
    assert len(lasp_configs) == len(potential_configs)
    
    if len(lasp_configs) == 1:
        task_generator = LaspTaskGeneration(
            lasp_config=lasp_configs[0], potential_config=potential_configs[0], whether_to_cover=whether_to_cover)

        for idx, structure in enumerate(structures, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = task_generator.get_task(task_dir=task_dir, structure=structure)

            task_dirs.append(task_dir)
            task_list.append(task)
        
    elif len(structures) == 1:

        for idx, configs in enumerate(zip(lasp_configs, potential_configs), start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = LaspTaskGeneration(lasp_config=configs[0], potential_config=configs[1], whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=structures[0])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    else:
        assert len(lasp_configs) == len(structures)

        for idx, configs_structure in enumerate(zip(lasp_configs, potential_configs, structures), start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = LaspTaskGeneration(
                lasp_config=configs_structure[0],
                potential_config=configs_structure[1],
                whether_to_cover=whether_to_cover
            ).get_task(task_dir=task_dir, structure=configs_structure[2])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)