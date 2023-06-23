import glob
from pathlib import Path
from typing import Union

from ..runtask.run import RunTask
from ..runtask.generator import GeneralUserConfig
from ..runtask.generator.lammps import lammps_task_factory



def check_lammps_input(fpath:str):

    simples = Path(fpath).glob("**/in.simple")
    assert simples != []


def lammps_flow(
    user_config:Union[dict, GeneralUserConfig],
    structures: list,
    output_dir:Union[str, Path]=".", 
    whether_to_cover:bool=False,
    whether_to_return:bool=False):

    user_config = GeneralUserConfig.parse_obj(user_config)

    if "models" in user_config.params:
        models = glob.glob(user_config.params["models"]) if isinstance(user_config.params["models"], str) else user_config.params["models"]
        assert isinstance(models, list)
        temp_models = []
        for model in models:
            if model not in user_config.dpdispatcher.forward_common_files:
               user_config.dpdispatcher.forward_common_files.append(model)
               temp_models.append(f"../{Path(model).name}")
        user_config.params["models"] = temp_models

    tasks = lammps_task_factory(
        lammps_configs=[user_config],
        structures=structures,
        output_dir=output_dir,
        whether_to_cover=whether_to_cover)

    runner = RunTask(
        output_dir=output_dir,
        dpdispatcher_config=user_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)


    check_lammps_input(fpath=output_dir)
    
    runner.run_submission()

    if whether_to_return:
        return tasks