import json, time
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

from ase.io import read
from pydantic import BaseModel

from .base import RecordLog, WorkflowGeneration, InitialStructConfig, InitialStruct
from .. import wflog
from ..runtask.generator import GeneralUserConfig
from ..pretask.sets import load_config, update_config_by_accuracy
from ..unittask.fp import vasp_flow, cp2k_flow
from ..unittask.dp import dptrain_flow, model_devi_percent, DPTrainFlowConfig
from ..unittask.ssw import lammps_ssw_flow
from ..utils.utils import get_dir_percent_dict, get_accuracy_candidate_mean
from ..posttask.process import DatasetGeneration, CandidateProcess, ModelDeviProcess
from ..posttask.plot import ModelDeviPlot


class EndCriterionConfig(BaseModel):
    accuracy: float
    mean_devi: Optional[float] = None
    max_devi: Union[float, int] = 1
    numb_reach: int = 2


class GenMlpsConfig(BaseModel):
    class General(BaseModel):
        dataset: str
        type_map: List[str]
        end_criterion: EndCriterionConfig
        bad_data: Optional[str] = None
        init: Optional[InitialStructConfig]=None

        system_prefix: str="system"
        train_dirname: str="00.dp_train"
        explore_dirname: str="01.dp_lasp"
        label_dirname: str="02.fp_label"
    
    class Training(BaseModel):
        numb_train: int = 4
        dp: GeneralUserConfig

    class Exploration(BaseModel):
        lasp: GeneralUserConfig
        lammps: GeneralUserConfig = GeneralUserConfig.parse_obj({})
        numb_struct_per_system: int = 1
    
    class Screening(BaseModel):
        bounds: List[float]
        numb_candidate_per_traj: int = 25
        asap: Optional[dict] = None
        numb_struct_per_label: int = 1
        noise_percent: Union[int, float] = 100

    class Labeling(BaseModel):
        f_cutoff: Union[float, int, None] = 10
        vasp: Optional[GeneralUserConfig]=None
        cp2k: Optional[GeneralUserConfig]=None

    class AccuracyUpdate(BaseModel):
        accuracy: List[float]
        new_config: Union[List[dict], List[str]]

    general: General
    training: Training
    exploration: Exploration
    screening: Screening
    labeling: Labeling
    update_config: Optional[AccuracyUpdate]=None



def get_itername(iter_idx:int):

    return Path("iter.%06d" % iter_idx)


## dp training
def training(user_config:GenMlpsConfig, iter_idx:int):

    config = DPTrainFlowConfig.parse_obj({
        "numb_train": user_config.training.numb_train,
        "dataset": user_config.general.dataset,
        "type_map": user_config.general.type_map,
        "bad_data": user_config.general.bad_data,
        "dp_config": user_config.training.dp
    })

    dptrain_flow(
        user_config=config,
        output_dir=get_itername(iter_idx=iter_idx)/user_config.general.train_dirname
    )


## dp ssw exploration
def exploration(user_config:GenMlpsConfig, iter_idx:int):

    structs = ssw_structs(user_config=user_config, iter_idx=iter_idx)

    modified_lammps_config(user_config=user_config, iter_idx=iter_idx)
    config = {"lasp_config": user_config.exploration.lasp, "potential_config": user_config.exploration.lammps}

    lammps_ssw_flow(
        user_config=config,
        structures=structs,
        output_dir=get_itername(iter_idx=iter_idx)/user_config.general.explore_dirname
    )


def ssw_structs(user_config:GenMlpsConfig, iter_idx:int):

    dataset = user_config.general.dataset
    system_prefix = user_config.general.system_prefix
    init = user_config.general.init
    numb_struct_per_system = user_config.exploration.numb_struct_per_system

    if iter_idx > 0 and Path(dataset).glob(f"{system_prefix}-{iter_idx-1}-*") != []:
        ms = DatasetGeneration.get_dataset(dirname=dataset, filename=f"{system_prefix}-{iter_idx-1}-*")
        candidate_structs = DatasetGeneration.random_struct(ms=ms, numb_struct_per_system=numb_struct_per_system) 
    
    init_structs = [] if init is None else InitialStruct(config=init).collect_structures()
    ms = DatasetGeneration.get_extra_dataset(dirname=dataset, common_str=system_prefix) 
    try:
        ssw_structs = DatasetGeneration.random_struct(ms=ms, numb_struct_per_system=numb_struct_per_system) + init_structs + candidate_structs
    except:
        ssw_structs = DatasetGeneration.random_struct(ms=ms, numb_struct_per_system=numb_struct_per_system) + init_structs
    
    return ssw_structs


def modified_lammps_config(user_config:GenMlpsConfig, iter_idx:int):

    lammps_config = user_config.exploration.lammps
    lammps_config.params["type_map"] = user_config.general.type_map 

    if lammps_config.template_path is None:
        lammps_config.params["models"] = str(
            get_itername(iter_idx=iter_idx)/user_config.general.train_dirname/"train.*/*.pb")
    
    user_config.exploration.lammps = lammps_config


## model_devi and cluster screening
def screening(user_config:GenMlpsConfig, iter_idx:int):

    devi_dirpaths = sorted((get_itername(iter_idx=iter_idx)/user_config.general.explore_dirname).glob("task.*"))
    bounds = user_config.screening.bounds

    tot_percent, percents = model_devi_percent(
        devi_dirpaths=devi_dirpaths, 
        bounds=bounds,
        max_devi=user_config.general.end_criterion.max_devi,
        )

    output_dir=get_itername(iter_idx=iter_idx)
    with open(str(output_dir/"percents.json"), "w") as f:
        json.dump(get_dir_percent_dict(dirpaths=devi_dirpaths, percents=percents), f, indent=4)

    with open(str(output_dir/"tot_percent.json"), "w") as f:
        json.dump(tot_percent, f, indent=4)
    
    if iter_idx > 0:
        model_devi_plot(dirname=Path(user_config.general.explore_dirname)/"task.*", bounds=bounds, iter_idx=iter_idx)
    
    ssw_output = True if user_config.exploration.lasp.params.get("SSW.output") == "T" else False
    CandidateProcess(
        dirpaths=list((dirpath for idx, dirpath in enumerate(devi_dirpaths) if percents[idx]["accuracy"] < user_config.general.end_criterion.accuracy)),
        output_file=output_dir/"candidates.xyz",
        bounds=user_config.screening.bounds,
        ssw_output=ssw_output
    ).filing_chosen_candidates(
        numb_candidate_per_traj=user_config.screening.numb_candidate_per_traj,
        whether_to_cluster=True,
        asap_config=user_config.screening.asap,
        numb_struct_per_label=user_config.screening.numb_struct_per_label,
        noise_percent=user_config.screening.noise_percent) 


def model_devi_plot(dirname:Union[str,Path], bounds:List[float], iter_idx:int):

    iter_list = list(range(iter_idx+1))
    accuracy, candidate, mean = get_accuracy_candidate_mean(filespath="iter.*/tot_percent.json")
    model_devis = list((
        ModelDeviProcess(dirpath=get_itername(iter_idx=i).joinpath(dirname), bounds=bounds).get_max_force_devi() for i in range(iter_idx+1)))

    plotclass = ModelDeviPlot(output_dir=str(get_itername(iter_idx=iter_idx)), bounds=bounds)
    plotclass.plot_accuracy_bar(iter_list=iter_list, accuracy=accuracy, candidate=candidate, means=mean)
    plotclass.plot_distribution(iter_list=iter_list, model_devis=model_devis)


## end criterion
def end_criterion(user_config:GenMlpsConfig, iter_idx:int, wf=WorkflowGeneration):

    numb_reach = user_config.general.end_criterion.numb_reach
    mean_devi = user_config.general.end_criterion.mean_devi 

    if iter_idx >= numb_reach:
        filespath = list((get_itername(iter_idx=i).joinpath("tot_percent.json") for i in range(iter_idx+1)[-(numb_reach+1):]))
        accuracy, _, mean = get_accuracy_candidate_mean(filespath=filespath)

        if (np.array(accuracy[-numb_reach:]) >= user_config.general.end_criterion.accuracy).all():
            wf.set_cont(cont=False)
            wflog.info("-" * 75)
            wflog.info("This workflow of MLPs generation reaches the accuracy end_criterion and successfully finishes!")
            wflog.info("-" * 75)
        
        elif mean_devi is not None and (np.array(list(mean[i+1]-mean[i] for i in range(numb_reach))) < mean_devi).all():
            wf.set_cont(cont=False)
            wflog.info("-" * 75)
            wflog.info("This workflow of MLPs generation reaches the mean_devi end_criterion and successfully finishes!")
            wflog.info("-" * 75)
    

## fp labeling
def labeling(user_config:GenMlpsConfig, iter_idx:int):

    output_dir = get_itername(iter_idx=iter_idx)
    structures = read(output_dir/"candidates.xyz", ":", format="extxyz")
    label_config = user_config.labeling
    assert not(label_config.cp2k is None and label_config.vasp is None)

    if len(structures) != 0:
        if label_config.cp2k is not None:
            cp2k_flow(
                user_config=label_config.cp2k,
                structures=structures,
                output_dir=output_dir/user_config.general.label_dirname
            )    
        elif label_config.vasp is not None:
            vasp_flow(
                user_config=label_config.vasp,
                structures=structures,
                output_dir=output_dir/user_config.general.label_dirname
            )       
        else:
            raise KeyError("The key word of first principle style is wrong!")
    else:
        wflog.info(f"The number of candidates is zero in the iteriation {iter_idx} and pass the labling task.")


def add_data(user_config:GenMlpsConfig, iter_idx:int):
    
    label_config = user_config.labeling
    dirname = get_itername(iter_idx=iter_idx)/user_config.general.label_dirname/"task.*"
    filename, fmt = ("OUTCAR", "vasp/outcar") if label_config.cp2k is None else ("cp2k.out", "cp2k/output")
    ms = DatasetGeneration.get_dataset(dirname=dirname, filename=filename, fmt=fmt, type_map=user_config.general.type_map)

    DatasetGeneration(
        dataset=user_config.general.dataset,
        bad_data=user_config.general.bad_data,
        system_prefix=user_config.general.system_prefix
    ).gen_dataset(ms=ms, iter_idx=iter_idx, cutoff=label_config.f_cutoff)


## accuracy-based updating
def update_config(user_config:GenMlpsConfig, iter_idx:int):
    
    if user_config.update_config is not None:
        with open(str(get_itername(iter_idx=iter_idx)/"tot_percent.json"), "r", encoding="utf8") as f:
            accuracy = json.load(f)["accuracy"]

        new_config = update_config_by_accuracy(
            accuracy=accuracy,
            user_config=user_config.dict(),
            iter_idx=iter_idx)

        return GenMlpsConfig.parse_obj(new_config)
    
    else:
        return user_config



def gen_mlps_flow(user_config:Union[GenMlpsConfig, str, dict], max_iter:int=30, wf=WorkflowGeneration):

    user_config = GenMlpsConfig.parse_obj(user_config) if not isinstance(user_config, str) \
        else GenMlpsConfig.parse_obj(load_config(user_config))
    
    task_list = [training, exploration, screening, end_criterion, labeling, add_data]

    recordlog = RecordLog(record="gen_mlps.record")
    iter_idx, task_idx = recordlog.check_checkpoint_file()
    task_idx += 1

    gen_mlps = wf(config={
        "task_idx": task_idx,
        "iter_idx": iter_idx,
        "task_list": task_list,
        "max_iter": max_iter
    })
    
    while wf.get_cont():

        iter_idx, task_idx = gen_mlps.get_idxes
        
        output_dir = get_itername(iter_idx=iter_idx)
        if not output_dir.is_dir():
            Path.mkdir(output_dir)

        if iter_idx==0 and task_idx==0:
            wflog.info("-" * 75)
            wflog.info(time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime()))
            wflog.info("Machine Learning Potential Generation:")
            wflog.info(
                f"Training (DeePMD-kit) --> Exploration (LASP-SSW) --> Labelling(FP)")
            wflog.info("-" * 75)

        gen_mlps.run_workflow(
            record=recordlog, 
            user_config=user_config,  
            iter_idx=iter_idx)

        user_config = update_config(user_config=user_config, iter_idx=iter_idx)

    
    
