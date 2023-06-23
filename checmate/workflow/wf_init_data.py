import time
from pathlib import Path
from typing import Union, Optional, List

from pydantic import BaseModel

from .base import RecordLog, InitialStruct, InitialStructConfig, WorkflowGeneration
from .. import wflog
from ..pretask.sets import load_config
from ..runtask.generator import GeneralUserConfig
from ..posttask.process import DatasetGeneration
from ..unittask.fp import vasp_flow, cp2k_flow
from ..unittask.ssw import fp_ssw_flow


class InitDataConfig(BaseModel):
    class General(BaseModel):
        dataset: str
        init: InitialStructConfig
        type_map: Optional[List[str]] = None
        bad_data: Optional[str] = None      

        explore_dirname: str="00.ssw_pre_sample"
        label_dirname: str="01.fp_label"

    class Exploration(BaseModel):
        lasp: GeneralUserConfig
        vasp: Optional[GeneralUserConfig] = GeneralUserConfig.parse_obj({})
        cp2k: Optional[GeneralUserConfig] = GeneralUserConfig.parse_obj({})

    class Labeling(BaseModel):
        numb_structs: int = 200
        f_cutoff: Union[float, int, None] = 10
        vasp: Optional[GeneralUserConfig]=None
        cp2k: Optional[GeneralUserConfig]=None

    general: General
    exploration: Optional[Exploration]=None
    labeling: Labeling

## fp ssw exploration
def pre_sampling(user_config:InitDataConfig, output_dir:Union[str,Path]):

    structures = InitialStruct(config=user_config.general.init).collect_structures()
    potential = user_config.exploration.lasp.params.get("potential", "vasp")
    if potential == "vasp":
        config = {"lasp_config": user_config.exploration.lasp, "potential_config": user_config.exploration.vasp}

    elif potential == "cp2k":
        config = {"lasp_config": user_config.exploration.lasp, "potential_config": user_config.exploration.cp2k}

    else:
        raise KeyError("The key word of first principle style is wrong!")

    fp_ssw_flow(
        user_config=config,
        structures=structures,
        output_dir=Path(output_dir)/user_config.general.explore_dirname)


## fp labeling
def labeling(user_config:InitDataConfig, output_dir:Union[str,Path]):

    ssw_dir = Path(output_dir)/user_config.general.explore_dirname  

    if ssw_dir.is_dir():
        paths = list(ssw_dir.glob("task.*/allstr.arc"))
        structures =  InitialStruct(
            config={
                "structure_paths": paths,
                "structure_format": "dmol-arc", 
                "numb_struct_per_file": int(user_config.labeling.numb_structs/len(paths)),
        }).collect_structures()
    else:
        structures = InitialStruct(config=user_config.general.init).collect_structures()
    
    assert len(structures) != 0

    label_config  = user_config.labeling
    if label_config.cp2k is not None:
        cp2k_flow(
            user_config=label_config.cp2k,
            structures=structures,
            output_dir=Path(output_dir)/user_config.general.label_dirname
        )    
    elif label_config.vasp is not None:
        vasp_flow(
            user_config=label_config.vasp,
            structures=structures,
            output_dir=Path(output_dir)/user_config.general.label_dirname
        )       
    else:
        raise KeyError("The key word of first principle style is wrong!")


## init dataset generation
def gen_dataset(user_config:dict, output_dir:Union[str,Path]):

    label_config = user_config.labeling
    dirname = Path(output_dir)/user_config.general.label_dirname/"task.*"
    filename, fmt = ("OUTCAR", "vasp/outcar") if label_config.cp2k is None else ("cp2k.out", "cp2k/output")
    ms = DatasetGeneration.get_dataset(dirname=dirname, filename=filename, fmt=fmt, type_map=user_config.general.type_map)

    DatasetGeneration(
        dataset=user_config.general.dataset,
        bad_data=user_config.general.bad_data
    ).gen_dataset(ms=ms, cutoff=label_config.f_cutoff)



def init_data_flow(user_config:Union[InitDataConfig,dict,str], whether_to_ssw:bool=True, wf=WorkflowGeneration):

    user_config = InitDataConfig.parse_obj(user_config) if not isinstance(user_config, str) \
        else InitDataConfig.parse_obj(load_config(user_config))
    
    output_dir = Path("init.ssw_fp") if whether_to_ssw else Path("init.fp")
    process_name = "LASP-FP" if whether_to_ssw else "FP"
    task_list = [pre_sampling, labeling, gen_dataset] if whether_to_ssw else [labeling, gen_dataset]

    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    recordlog = RecordLog(dirpath=output_dir.parent, record="init_data.record")
    iter_idx, task_idx = recordlog.check_checkpoint_file()
    task_idx += 1

    init_data = wf(config={
        "task_idx": task_idx,
        "iter_idx": iter_idx,
        "task_list": task_list,
        "max_iter": 1
    })

    while iter_idx == 0:

        if task_idx==0:
            wflog.info("-" * 75)
            wflog.info(time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime()))
            wflog.info(f"Dataset Initializaton: {process_name}")
            wflog.info("-" * 75)
            wflog.info("Start Collecting Initial Dataset:")

        init_data.run_workflow(
            record=recordlog, 
            user_config=user_config,  
            output_dir=output_dir)

        iter_idx, task_idx = init_data.get_idxes
    
        wflog.info("Finish Collecting Initial Dataset!")
        wflog.info('-' * 75)
