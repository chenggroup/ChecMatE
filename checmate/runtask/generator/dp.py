import copy
from pathlib import Path
from os.path import abspath
from typing import List, Optional, Union

from pydantic import BaseModel
from dpdispatcher.submission import Task

from . import GeneralUserConfig, BaseTaskGeneration, GeneralFactaryOutput
from ... import wflog
from ...utils.utils import dp_type_map
from ...pretask.sets import update_dict
from ...pretask.inputs import DPTrainInput



# DP Modules
class DPTaskConfig(BaseModel):
    command: str = "dp"
    forward_files: List[str] = ["input.json"]
    backward_files: List[str] = ["dp.err", "dp.log", "out.json", "model.ckpt.data-00000-of-00001", 
                                 "model.ckpt.index", "model.ckpt.meta", "lcurve.out"]
    errlog: str = "dp.err"
    outlog: str = "dp.log"


class DPTrainTaskGeneration(BaseTaskGeneration):

    def __init__(
        self,
        dp_config: Union[dict, GeneralUserConfig],
        dataset: str,
        type_map: List[str],
        bad_data: Optional[str]=None,
        whether_to_cover: bool=False       
    ):  
        self.dp_config = copy.deepcopy(GeneralUserConfig.parse_obj(dp_config))
        self.whether_to_cover = whether_to_cover

        self.dataset = abspath(dataset)
        self.type_map = type_map
        self.bad_data = bad_data
        self.systems = []

    
    def __update_params_config_for_systems(self, task_dir:str):

        task_dir = Path(abspath(task_dir))
        params_config = self.dp_config.params

        dataset = Path(self.dataset)
        if not (task_dir.parent/dataset.name).is_dir():
            (task_dir.parent/dataset.name).symlink_to(dataset)

        type_raws = list(dataset.glob("**/type.raw"))
        systems = list((f"../{dataset.name}/{i.parent.name}") for i in type_raws)
        assert systems != []

        if (type_raws[0].parent/"type_map.raw").is_file():
            data_type_map = dp_type_map(str(type_raws[0].parent/"type_map.raw"))
            if self.type_map != data_type_map:
                wflog.info(f"WARNING: type_map {self.type_map} given by user can not match data_type_map {data_type_map} read from dataset!")
        
        if self.bad_data is not None and Path(self.bad_data).is_dir():
            bad_data = Path(abspath(self.bad_data))
            if not (task_dir.parent/bad_data.name).is_dir():
                (task_dir.parent/bad_data.name).symlink_to(bad_data)

            lm = len(systems)
            systems += list((f"../{bad_data.name}/{i.parent.name}") for i in bad_data.glob("**/type.raw"))

            if len(systems) > lm:
                update_dict(
                    params_config, 
                    {"training":{
                        "training_data":{
                            "auto_prob": f"prob_sys_size;0:{lm}:0.9995;{lm}:{len(systems)}:0.0005"}}})
        
        self.dp_config.params = params_config
        self.systems = systems


    def gen_task_inputs(self, task_dir:str, inputclass=DPTrainInput):

        self.__update_params_config_for_systems(task_dir=task_dir)
        assert self.systems != []

        inputfile = Path(task_dir)/"input.json"
        template_path = self.dp_config.template_path
        if template_path:
            inputs = inputclass(
                systems=self.systems,
                type_map=self.type_map,
                user_config=self.dp_config.params,
                template_path=template_path)
        else:
            inputs = inputclass(
                systems=self.systems,
                type_map=self.type_map,
                user_config=self.dp_config.params)

        if not inputfile.exists():
            inputs.write_input(output_dir=task_dir)
        else:
            wflog.info("The input files of dp train task has already existed! Please confirm whether overwrite. By default, it is false.")
            if self.whether_to_cover:
                inputs.write_input(output_dir=task_dir)

        
    def get_task(self, task_dir:str):

        self.gen_task_inputs(task_dir=task_dir)

        task_config = DPTaskConfig.parse_obj(self.dp_config.dpdispatcher.task)

        try:
            idx = int(Path(task_dir).suffix.strip("."))
            potential_fname = f"frozen_model{idx}.pb"
        except:
            potential_fname = "frozen_model.pb"

        command = task_config.command
        task_config.command = command.join([
            "", " train input.json && ", f" freeze -o {potential_fname} && ", f" test -m {potential_fname} -s ../{dataset.name}"])
        
        return Task(
                    task_work_path=Path(task_dir).name,
                    **(task_config.dict())
                )


class DPTestTaskGeneration(BaseTaskGeneration):

    def __init__(
        self,
        dp_config: Union[dict, GeneralUserConfig],
        dataset: str,
        whether_to_cover: bool=False      
    ):  
        self.dp_config = GeneralUserConfig.parse_obj(dp_config)
        self.whether_to_cover = whether_to_cover

        self.dataset = abspath(dataset)

    def gen_task_inputs(self, task_dir:str, model_path:str, make_dir_if_not_present:bool=True):

        task_dir = Path(abspath(task_dir))
        if make_dir_if_not_present and not task_dir.is_dir():
            Path.mkdir(task_dir)

        model_path = Path(abspath(model_path))
        assert model_path.is_file()
        des = task_dir.joinpath(model_path.name)
        
        try:
            des.symlink_to(model_path)
        except:
            des.unlink()
            des.symlink_to(model_path)


    def get_task(self, task_dir:str, model_path:str):

        dataset = Path(self.dataset)
        task_dir = Path(abspath(task_dir))
        if not (task_dir.parent/dataset.name).is_dir():
            (task_dir.parent/dataset.name).symlink_to(dataset)

        self.gen_task_inputs(task_dir=task_dir, model_path=model_path)
        model_path = Path(model_path)

        task_config = DPTaskConfig.parse_obj({
            "forward_files": [model_path.name],
            "backward_files": ["dp.err", "dp.log", "detail_file.e.out", "detail_file.f.out", "detail_file.v.out"]
        }).dict()
        update_dict(task_config, self.dp_config.dpdispatcher.task)

        task_config["command"] += f" test -m {model_path.name} -s ../{dataset.name} -d detail_file"

        return Task(
                    task_work_path=task_dir.name,
                    **task_config
                )



def dptrain_task_factory(
        dp_config: Union[dict, GeneralUserConfig], 
        numb_train: int, 
        dataset: str,
        type_map: List[str],
        bad_data: Optional[str]=None,
        output_dir:str=".", 
        whether_to_cover:bool=False):

    task_dirs, task_list = [], []
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    task_generator = DPTrainTaskGeneration(
        dp_config=dp_config,
        dataset=dataset,
        type_map=type_map,
        bad_data=bad_data, 
        whether_to_cover=whether_to_cover)

    for idx in range(1, numb_train+1):
        task_dir = output_dir.joinpath(f"train.{idx:02}")
        task = task_generator.get_task(task_dir=task_dir)

        task_dirs.append(task_dir)
        task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)


def dptest_task_factory(
        dp_config: Union[dict, GeneralUserConfig], 
        models: List[str], 
        dataset: str,
        output_dir:str=".", 
        whether_to_cover:bool=False):

    task_dirs, task_list = [], []
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)
    
    task_generator = DPTestTaskGeneration(
        dp_config=dp_config,
        dataset=dataset,
        whether_to_cover=whether_to_cover)

    for idx, model_path in enumerate(models, start=1):
        task_dir = output_dir.joinpath(f"test.{idx:02}")
        task = task_generator.get_task(task_dir=task_dir, model_path=model_path)

        task_dirs.append(task_dir)
        task_list.append(task)
    
    return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)