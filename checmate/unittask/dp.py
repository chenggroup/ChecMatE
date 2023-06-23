import copy, json
import numpy as np
from pathlib import Path
from typing import Union, Optional, List

from pydantic import BaseModel

from .. import wflog
from ..runtask.run import RunTask
from ..runtask.generator import GeneralUserConfig
from ..runtask.generator.dp import dptrain_task_factory, dptest_task_factory
from ..posttask.process import ModelDeviProcess


class DPTrainFlowConfig(BaseModel):
    numb_train: int = 4
    dataset: str
    type_map: List[str]
    bad_data: Optional[str] = None
    dp_config: GeneralUserConfig


class DPTestFlowConfig(BaseModel):
    models: List[str]
    dataset: str
    dp_config: GeneralUserConfig


def check_dptrain_input(fpath:Union[str, Path], dataset:Union[str, Path]):

    inputs = Path(fpath).glob("**/input.json")
    assert inputs != []
    assert Path(dataset).is_dir()


def check_dptest_input(fpath:Union[str, Path], dataset:Union[str, Path]):

    inputs = Path(fpath).glob("**/*.pb")
    assert inputs != []
    assert Path(dataset).is_dir()


def dptrain_flow(
    user_config:Union[dict, DPTrainFlowConfig], 
    output_dir:Union[str, Path]=".", 
    whether_to_cover:bool=False, 
    whether_to_return:bool=False):

    user_config = copy.deepcopy(DPTrainFlowConfig.parse_obj(user_config))

    tasks = dptrain_task_factory(
        **(user_config.dict()),
        output_dir=output_dir,
        whether_to_cover=whether_to_cover
    )

    dataset = user_config.dataset
    if dataset not in user_config.dp_config.dpdispatcher.forward_common_files:
        user_config.dp_config.dpdispatcher.forward_common_files.append(dataset)

    runner = RunTask(
            output_dir=output_dir,
            dpdispatcher_config=user_config.dp_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)

    check_dptrain_input(fpath=output_dir, dataset=user_config.dataset)
    runner.run_submission()

    if whether_to_return:
        return tasks


def dptest_flow(
    user_config:Union[dict, DPTestFlowConfig], 
    output_dir:Union[str, Path]=".", 
    whether_to_cover:bool=False, 
    whether_to_return:bool=False):

    user_config = copy.deepcopy(DPTestFlowConfig.parse_obj(user_config))

    tasks = dptest_task_factory(
        **(user_config.dict()),
        output_dir=output_dir,
        whether_to_cover=whether_to_cover
    )

    dataset = user_config.dataset
    if dataset not in user_config.dp_config.dpdispatcher.forward_common_files:
        user_config.dp_config.dpdispatcher.forward_common_files.append(dataset)

    runner = RunTask(
            output_dir=output_dir,
            dpdispatcher_config=user_config.dp_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)
    
    check_dptest_input(fpath=output_dir, dataset=user_config.dataset)
    runner.run_submission()

    if whether_to_return:
        return tasks



def model_devi_percent(devi_dirpaths:Union[List[str], List[Path]], bounds:List[float], max_devi:Union[float,int]=1):

    max_force_devis = list((np.loadtxt(Path(i)/"model_devi.out"))[:, 4] for i in devi_dirpaths)
    tot_max_force_devi = np.array([j for i in max_force_devis for j in i])

    taskclass = ModelDeviProcess(bounds=bounds)
    tot_percent = taskclass.get_devi_percent(max_force_devi=tot_max_force_devi, max_devi=max_devi)
    percents = list(taskclass.get_devi_percent(max_force_devi=i, max_devi=max_devi) for i in max_force_devis)

    wflog.info(f"The total accuracy of this iteration is:  {tot_percent['accuracy']:0.3f}.")
    wflog.info(f"The max_force_devi of this iteration is:  {tot_max_force_devi.max():0.3f}.")
    wflog.info(f"The mean of model_devi in this iteration is: {tot_percent['mean']:0.3f}.") 

    return tot_percent, percents
    
    
