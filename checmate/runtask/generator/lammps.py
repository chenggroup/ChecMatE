import copy
from pathlib import Path
from os.path import abspath
from typing import List, Union

from ase.atoms import Atoms
from pydantic import BaseModel
from dpdispatcher.submission import Task

from . import GeneralUserConfig, BaseTaskGeneration, GeneralFactaryOutput
from ... import wflog
from ...pretask.inputs import LammpsInput


#LAMMPS Modules
class LammpsTaskConfig(BaseModel):
    command: str = "lmp_mpi"
    forward_files: List[str] = ["in.simple", "data.simplr"]
    backward_files: List[str] = ["lammps.err", "lammps.log"]
    errlog: str = "lammps.err"
    outlog: str = "lammps.log"


class LammpsTaskGeneration(BaseTaskGeneration):

    def __init__(
        self,
        lammps_config: Union[dict, GeneralUserConfig],
        whether_to_cover: bool=False
    ):
        self.lammps_config = copy.deepcopy(GeneralUserConfig.parse_obj(lammps_config))
        self.whether_to_cover = whether_to_cover


    def gen_task_inputs(self, task_dir:str, structure:Atoms, inputclass=LammpsInput):

        inputfile = (Path(task_dir)/"in.simple")
        template_path = self.lammps_config.template_path
        if template_path:
            inputs = inputclass(
                structure=structure,
                user_config=self.lammps_config.params,
                template_path=template_path)

        else:
            inputs = inputclass(
                structure=structure,
                user_config=self.lammps_config.params)

        if not inputfile.exists():
            inputs.write_input(output_dir=task_dir)
        else:
            wflog.info("The input files of lammps task has already existed! Please confirm whether overwrite. By default, it is false.")
            if self.whether_to_cover:
                inputs.write_input(output_dir=task_dir) 


    def get_task(self, task_dir:str, structure:Atoms):

        self.gen_task_inputs(task_dir=task_dir, structure=structure)

        task_config = LammpsTaskConfig.parse_obj(self.lammps_config.dpdispatcher.task)
        task_config.command += " -i in.simple" 
        
        return Task(
                    task_work_path=Path(task_dir).name,
                    **(task_config.dict())
                )



def lammps_task_factory(
        lammps_configs: Union[List[dict], List[GeneralUserConfig], dict, GeneralUserConfig], 
        structures: List[Atoms], 
        output_dir:str=".", 
        whether_to_cover:bool=False):

    lammps_configs = [lammps_configs] if not isinstance(lammps_configs, list) else lammps_configs
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    task_dirs, task_list = [], []
    assert len(lammps_configs) != 0 and len(structures) != 0
    
    if len(lammps_configs) == 1:
        task_generator = LammpsTaskGeneration(lammps_config=lammps_configs[0], whether_to_cover=whether_to_cover)

        for idx, structure in enumerate(structures, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = task_generator.get_task(task_dir=task_dir, structure=structure)

            task_dirs.append(task_dir)
            task_list.append(task)
        
    elif len(structures) == 1:

        for idx, lammps_config in enumerate(lammps_configs, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = LammpsTaskGeneration(lammps_config=lammps_config, whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=structures[0])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    else:
        assert len(lammps_configs) == len(structures)

        for idx, config_structure in enumerate(zip(lammps_configs, structures), start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = LammpsTaskGeneration(lammps_config=config_structure[0], whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=config_structure[1])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)
