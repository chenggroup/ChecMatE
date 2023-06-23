import copy
from pathlib import Path
from os.path import abspath
from typing import List, Union

from ase.atoms import Atoms
from pydantic import BaseModel
from dpdispatcher.submission import Task

from . import GeneralUserConfig, BaseTaskGeneration, GeneralFactaryOutput
from ... import wflog
from ...pretask.inputs import VaspStaticInput



#First-principle VASP Modules
class VaspTaskConfig(BaseModel):
    command: str = "mpiexec.hydra -genvall vasp_std"
    forward_files: List[str] = ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
    backward_files: List[str] = ["vasp.err", "vasp.log", "OUTCAR", "vasprun.xml", 
                                 "CONTCAR", "OSZICAR", "EIGENVAL", "CHGCAR"]
    errlog: str = "vasp.err"
    outlog: str = "vasp.log"


class VaspTaskGeneration(BaseTaskGeneration):

    def __init__(
        self,
        vasp_config: Union[dict, GeneralUserConfig],
        whether_to_cover: bool=False
    ):
        self.vasp_config = copy.deepcopy(GeneralUserConfig.parse_obj(vasp_config))
        self.whether_to_cover = whether_to_cover


    def gen_task_inputs(self, task_dir:str, structure:Atoms, inputclass=VaspStaticInput):

        inputfile = (Path(task_dir)/"INCAR")
        template_path = self.vasp_config.template_path
        if template_path:
            inputs = inputclass(
                structure=structure,
                user_config=self.vasp_config.params,
                template_path=template_path)

        else:
            inputs = inputclass(
                structure=structure,
                user_config=self.vasp_config.params)

        if not inputfile.exists():
            inputs.write_input(output_dir=task_dir)
        else:
            wflog.info("The input files of vasp task has already existed! Please confirm whether overwrite. By default, it is false.")
            if self.whether_to_cover:
                inputs.write_input(output_dir=task_dir) 


    def get_task(self, task_dir:str, structure:Atoms):

        self.gen_task_inputs(task_dir=task_dir, structure=structure)

        task_config = VaspTaskConfig.parse_obj(self.vasp_config.dpdispatcher.task)
        
        return Task(
                    task_work_path=Path(task_dir).name,
                    **(task_config.dict())
                )


    def gen_task_inputs_from_pre(self, task_dir:str, fpath:str, mode:str="line", line_density:int=20):

        from ase.io import read

        contcar = Path(abspath(fpath))/"CONTCAR"
        structure = read(str(contcar), format="vasp") if contcar.is_file() else read(str(contcar.parent/"POSCAR"), format="vasp")

        files_to_transfer = VaspStaticInput.from_pre_calc(fpath, self.vasp_config.params.get("files_to_transfer", {}))
        self.vasp_config.params["files_to_transfer"] = files_to_transfer

        if mode == "line":
            self.__line_mode_inputs(
                structure=structure, 
                task_dir=task_dir, 
                line_density=line_density)

        else:
            self.__uniform_mode_inputs(
                structure=structure, 
                task_dir=task_dir)


    def __line_mode_inputs(self, structure:Atoms, task_dir:str, line_density:int=20):

        from ...utils.tools.pmg import KpointsPath
        from ...pretask.sets import update_dict

        self.vasp_config.params["kpoints"] = KpointsPath(structure).gen_kpoints(line_density=line_density)

        incar_setting = {"LCHARG":False, "ICHARG":11, "LORBIT":11, "ISMEAR":0, "ISYM":0, "NSW":0}
        update_dict(incar_setting, self.vasp_config.params.get("incar", {}))
        self.vasp_config.params["incar"] = incar_setting

        self.gen_task_inputs(task_dir=task_dir, structure=structure)


    def __uniform_mode_inputs(self, structure:Atoms, task_dir:str, vasp_params:dict):

        from ...pretask.sets import update_dict

        incar_setting = {"LCHARG":False, "ICHARG":11, "LORBIT":11, "ISMEAR":-5, "NSW":0}
        update_dict(incar_setting, self.vasp_config.params.get("incar", {}))
        self.vasp_config.params["incar"] = incar_setting

        self.gen_task_inputs(task_dir=task_dir, structure=structure)


    def get_task_from_pre(self, task_dir:str, fpath:str, mode:str="line", line_density:int=20):

        self.gen_task_inputs_from_pre(task_dir=task_dir, fpath=fpath, mode=mode, line_density=line_density)

        files_to_transfer = self.vasp_config.params["files_to_transfer"]
        task_config = VaspTaskConfig.parse_obj(self.vasp_config.dpdispatcher.task)
        task_config.forward_files = list(set(task_config.forward_files + list(files_to_transfer)))  
        
        return Task(
                    task_work_path=Path(task_dir).name,
                    **(task_config.dict())
                )
    


def vasp_task_factory(
        vasp_configs: Union[List[dict], List[GeneralUserConfig], dict, GeneralUserConfig], 
        structures: List[Atoms], 
        output_dir:str=".", 
        whether_to_cover:bool=False):

    vasp_configs = [vasp_configs] if isinstance(vasp_configs, dict) else vasp_configs
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    task_dirs, task_list = [], []
    assert len(vasp_configs) != 0 and len(structures) != 0
    
    if len(vasp_configs) == 1:
        task_generator = VaspTaskGeneration(vasp_config=vasp_configs[0], whether_to_cover=whether_to_cover)

        for idx, structure in enumerate(structures, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = task_generator.get_task(task_dir=task_dir, structure=structure)

            task_dirs.append(task_dir)
            task_list.append(task)
        
    elif len(structures) == 1:

        for idx, vasp_config in enumerate(vasp_configs, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = VaspTaskGeneration(vasp_config=vasp_config, whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=structures[0])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    else:
        assert len(vasp_configs) == len(structures)

        for idx, config_structure in enumerate(zip(vasp_configs, structures), start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = VaspTaskGeneration(vasp_config=config_structure[0], whether_to_cover=whether_to_cover).get_task(
                task_dir=task_dir, structure=config_structure[1])
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)



def vasp_task_factory_from_pre(
        vasp_configs: Union[List[dict], List[GeneralUserConfig], dict, GeneralUserConfig], 
        fpaths: List[Atoms], 
        output_dir:str=".", 
        whether_to_cover:bool=False, 
        **kwargs):

    vasp_configs = [vasp_configs] if not isinstance(vasp_configs, list) else vasp_configs
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    task_dirs, task_list = [], []
    assert len(vasp_configs) != 0 and len(fpaths) != 0
    
    if len(vasp_configs) == 1:
        task_generator = VaspTaskGeneration(vasp_config=vasp_configs[0], whether_to_cover=whether_to_cover)

        for idx, fpath in enumerate(fpaths, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = task_generator.get_task_from_pre(task_dir=task_dir, fpath=fpath, **kwargs)

            task_dirs.append(task_dir)
            task_list.append(task)
        
    elif len(fpaths) == 1:

        for idx, vasp_config in enumerate(vasp_configs, start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = VaspTaskGeneration(vasp_config=vasp_config, whether_to_cover=whether_to_cover).get_task_from_pre(
                task_dir=task_dir, fpath=fpaths[0], **kwargs)
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    else:
        assert len(vasp_configs) == len(fpaths)

        for idx, config_fpath in enumerate(zip(vasp_configs, fpaths), start=1):
            task_dir = output_dir.joinpath(f"task.{idx:06}")
            task = VaspTaskGeneration(vasp_config=config_fpath[0], whether_to_cover=whether_to_cover).get_task_from_pre(
                task_dir=task_dir, structure=config_fpath[1], **kwargs)
            
            task_dirs.append(task_dir)
            task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)
