import glob
from pathlib import Path
from typing import Union

from pydantic import BaseModel

from .fp import check_vasp_input, check_cp2k_input
from .lammps import check_lammps_input
from ..runtask.run import RunTask
from ..runtask.generator import GeneralUserConfig
from ..runtask.generator.lasp import lasp_task_factory



def check_lasp_input(fpath:str):

    lasps = Path(fpath).glob("**/lasp.in")
    assert lasps != []


class LaspFlowConfig(BaseModel):
    lasp_config: GeneralUserConfig
    potential_config: GeneralUserConfig


def fp_ssw_flow(
    user_config:Union[dict, LaspFlowConfig], 
    structures:list, 
    output_dir:Union[str, Path]="./", 
    whether_to_cover:bool=False, 
    whether_to_return:bool=False):

    user_config = LaspFlowConfig.parse_obj(user_config)

    tasks = lasp_task_factory(
        lasp_configs=[user_config.lasp_config],
        potential_configs=[user_config.potential_config],
        structures=structures,
        output_dir=output_dir,
        whether_to_cover=whether_to_cover)
    
    runner = RunTask(
        output_dir=output_dir,
        dpdispatcher_config=user_config.lasp_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)


    potential = str.lower(user_config.lasp_config.params["potential"])

    if potential == "vasp":
        check_fp_input = check_vasp_input
    
    elif potential == "cp2k":
        check_fp_input = check_cp2k_input

    else:
        raise TypeError(f"The fp_style is not supported. Current support: vasp and cp2k.")

    check_fp_input(fpath=output_dir)
    check_lasp_input(fpath=output_dir)

    runner.run_submission()

    if whether_to_return:
        return tasks



def lammps_ssw_flow(
        user_config:Union[dict, LaspFlowConfig], 
        structures:list, output_dir:Union[str, Path]="./", 
        whether_to_cover:bool=False, 
        whether_to_return:bool=False):

    user_config = LaspFlowConfig.parse_obj(user_config)
    potential = str.lower(user_config.lasp_config.params.get("potential", "lammps"))
    assert potential == "lammps"

    if "models" in user_config.potential_config.params:
        models = user_config.potential_config.params["models"] 
        assert isinstance(models, (list, str))
        models = glob.glob(models) if isinstance(models, str) else models
        temp_models = []
        for model in models:
            if model not in user_config.lasp_config.dpdispatcher.forward_common_files:
               user_config.lasp_config.dpdispatcher.forward_common_files.append(model)
               temp_models.append(f"../{Path(model).name}")
        user_config.potential_config.params["models"] = temp_models

    tasks = lasp_task_factory(
        lasp_configs=[user_config.lasp_config],
        potential_configs=[user_config.potential_config],
        structures=structures,
        output_dir=output_dir,
        whether_to_cover=whether_to_cover)

    runner = RunTask(
        output_dir=output_dir,
        dpdispatcher_config=user_config.lasp_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)

    check_lammps_input(fpath=output_dir)
    check_lasp_input(fpath=output_dir)

    runner.run_submission()

    if whether_to_return:
        return tasks

