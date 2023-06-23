import copy
from pathlib import Path
from typing import List, Union

from ase.atoms import Atoms
from functools import partial
from pydantic import BaseModel
from dpdispatcher.submission import Task

from . import GeneralUserConfig, BaseTaskGeneration, GeneralFactaryOutput
from ... import wflog
from ...pretask.inputs import Cp2kStaticInput



#First-principle CP2K Modules
class Cp2kTaskConfig(BaseModel):
    command: str = "mpiexec.hydra cp2k.popt"
    forward_files: List[str] = ["cp2k.inp", "coord.xyz"]
    backward_files: List[str] = ["cp2k.err", "cp2k.log", "cp2k.out"]
    errlog: str = "cp2k.err"
    outlog: str = "cp2k.log"


class Cp2kTaskGeneration(BaseTaskGeneration):

    def __init__(
        self,
        cp2k_config: Union[dict, GeneralUserConfig],
        whether_to_cover: bool=False
    ):
        self.cp2k_config = copy.deepcopy(GeneralUserConfig.parse_obj(cp2k_config))
        self.whether_to_cover = whether_to_cover


    def gen_task_inputs(self, task_dir:Union[str, Path], structure:Atoms, inputclass=Cp2kStaticInput):

        inputfile = (Path(task_dir)/"cp2k.inp")
        template_path = self.cp2k_config.template_path
        if template_path:
            inputs = inputclass(
                structure=structure,
                user_config=self.cp2k_config.params,
                template_path=template_path)

        else:
            inputs = inputclass(
                structure=structure,
                user_config=self.cp2k_config.params)

        if not inputfile.exists():
            inputs.write_input(output_dir=task_dir)
        else:
            wflog.info("The input files of cp2k task has already existed! Please confirm whether overwrite. By default, it is false.")
            if self.whether_to_cover:
                inputs.write_input(output_dir=task_dir) 


    def get_task(self, task_dir:Union[str, Path], structure:Atoms) -> Task:

        self.gen_task_inputs(task_dir=task_dir, structure=structure)

        task_config = Cp2kTaskConfig.parse_obj(self.cp2k_config.dpdispatcher.task)
        task_config.command += " cp2k.inp >& cp2k.out" 
        
        return Task(
                    task_work_path=Path(task_dir).name,
                    **(task_config.dict())
                )



def cp2k_task_factory(
        cp2k_configs: Union[List[dict], List[GeneralUserConfig], dict, GeneralUserConfig], 
        structures: List[Atoms], 
        output_dir:str=".", 
        whether_to_cover:bool=False):

    cp2k_configs = [cp2k_configs] if not isinstance(cp2k_configs, list) else cp2k_configs
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    task_dirs, task_list = [], []
    assert len(cp2k_configs) != 0 and len(structures) != 0
    
    if len(cp2k_configs) == 1:
        task_generator = Cp2kTaskGeneration(cp2k_config=cp2k_configs[0], whether_to_cover=whether_to_cover)

        for idx, structure in enumerate(structures, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = task_generator.get_task(task_dir=task_dir, structure=structure)

            task_dirs.append(task_dir)
            task_list.append(task)
        
    elif len(structures) == 1:

        for idx, cp2k_config in enumerate(cp2k_configs, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = Cp2kTaskGeneration(cp2k_config=cp2k_config, whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=structures[0])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    else:
        assert len(cp2k_configs) == len(structures)

        for idx, config_structure in enumerate(zip(cp2k_configs, structures), start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = Cp2kTaskGeneration(cp2k_config=config_structure[0], whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=config_structure[1])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)

            
            

         
        


