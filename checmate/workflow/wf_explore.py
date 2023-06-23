import json, time
from pathlib import Path
from typing import Union, Optional, List

from ase.io import read, write
from pydantic import BaseModel

from .base import RecordLog, InitialStruct, InitialStructConfig, WorkflowGeneration
from .. import wflog
from ..pretask.sets import load_config
from ..runtask.generator import GeneralUserConfig
from ..unittask.dp import dptest_flow, model_devi_percent
from ..unittask.fp import vasp_flow, cp2k_flow
from ..unittask.ssw import lammps_ssw_flow
from ..utils.utils import get_dir_percent_dict 
from ..posttask.process import DatasetGeneration, ModelDeviProcess, ScreenProcess


class ExploreConfig(BaseModel):
    class General(BaseModel):
        dataset: str="test_data"
        type_map: List[str]
        init: InitialStructConfig
        bad_data: Optional[str] = None
        
        explore_dirname: str="00.dp_lasp"
        label_dirname: str="01.fp_label"
        test_dirname: str="02.dp_test"

    class Exploration(BaseModel):
        lasp: GeneralUserConfig
        lammps: GeneralUserConfig
    
    class Screening(BaseModel):
        e_cutoff: float = 0.1
        bounds: List[float] = [0.15, 0.3]
        asap: Optional[dict] = None
        numb_struct_per_label: int = 1
        noise_percent: Union[int, float] = 100

    class Labeling(BaseModel):
        f_cutoff: Union[float, int, None] = 10
        vasp: Optional[GeneralUserConfig] = None
        cp2k: Optional[GeneralUserConfig] = None
        dp: Optional[GeneralUserConfig] = None

    general: General
    exploration: Exploration
    screening: Screening
    labeling: Optional[Labeling] = None


## dp ssw exploration
def exploration(user_config:ExploreConfig, output_dir:str):

    structs = InitialStruct(config=user_config.general.init).collect_structures()

    user_config.exploration.lammps.params["type_map"] = user_config.general.type_map
    config = {"lasp_config": user_config.exploration.lasp, "potential_config": user_config.exploration.lammps}

    lammps_ssw_flow(
        user_config=config,
        structures=structs,
        output_dir=Path(output_dir)/user_config.general.explore_dirname
    )


## energy and cluster screening
def screening_by_e(user_config:ExploreConfig, output_dir:str):

    devi_dirpaths = sorted((Path(output_dir)/user_config.general.explore_dirname).glob("task.*"))
    if (devi_dirpaths[0]/"model_devi.out").is_file():
        model_devi(devi_dirpaths=devi_dirpaths, bounds=user_config.screening.bounds, output_dir=output_dir)
    
    structs = []
    for strfile in Path(output_dir).glob(f"{user_config.general.explore_dirname}/task.*/all.arc"):
        with open(str(strfile), "r") as f:
            energies = list((round(float(line[:73].strip().split()[-1]),6) for line in f.readlines() if "Energy" in line))

        structs.extend(ScreenProcess(
            structures=read(strfile, ":", "dmol-arc")
        ).run_filter_by_e(energies=energies, e_cutoff=user_config.screening.e_cutoff))
    
    write(str(Path(output_dir)/"screen_structs_by_e.xyz"), structs, format="extxyz")
    

def model_devi(devi_dirpaths:Union[List[str], List[Path]], bounds:List[float], output_dir:str, Devi=ModelDeviProcess):

    tot_percent, percents = model_devi_percent(
        devi_dirpaths=devi_dirpaths, 
        bounds=bounds)

    with open(str(Path(output_dir)/"percents.json"), "w") as f:
        json.dump(get_dir_percent_dict(dirpaths=devi_dirpaths, percents=percents), f, indent=4)

    with open(str(Path(output_dir)/"tot_percent.json"), "w") as f:
        json.dump(tot_percent, f, indent=4)


def screening_by_asap(user_config:ExploreConfig, output_dir:str, Screen=ScreenProcess):

    Screen.run_filter_by_cluster(
        fxyz=str(Path(output_dir)/"screen_structs_by_e.xyz"),
        asap_config=user_config.screening.asap,
        noise_percent=user_config.screening.noise_percent,
        numb_struct_per_label=user_config.screening.numb_struct_per_label,
        whether_to_write=True,
        whether_to_plot=True)
    

## fp labeling
def labeling(user_config:ExploreConfig, output_dir:str):

    structures = read(str(Path(output_dir)/"cluster_structs.xyz"), ":", "extxyz")
    assert len(structures) != 0

    label_config = user_config.labeling
    assert not(label_config.cp2k is None and label_config.vasp is None)

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


## dp test
def gen_test_data(user_config:ExploreConfig, output_dir:str):

    label_config = user_config.labeling
    dirname = Path(output_dir)/user_config.general.label_dirname/"task.*"
    filename, fmt = ("OUTCAR", "vasp/outcar") if label_config.cp2k is None else ("cp2k.out", "cp2k/output")
    ms = DatasetGeneration.get_dataset(dirname=dirname, filename=filename, fmt=fmt, type_map=user_config.general.type_map)

    DatasetGeneration(
        dataset=user_config.general.dataset,
        bad_data=user_config.general.bad_data
    ).gen_dataset(ms=ms, cutoff=label_config.f_cutoff)


def testing(user_config:ExploreConfig, output_dir:str):

    if user_config.labeling.dp:
        config = {
            "models": list((str(i) for i in Path(output_dir).glob(f"{user_config.general.explore_dirname}/*.pb"))),
            "dataset": user_config.general.dataset,
            "dp_config": user_config.labeling.dp
        }
        
        dptest_flow(
            user_config=config,
            output_dir=Path(output_dir)/user_config.general.test_dirname
        )


def explore_flow(user_config:Union[ExploreConfig,dict,str], whether_to_label:bool=True, wf=WorkflowGeneration):
    
    user_config = ExploreConfig.parse_obj(user_config) if not isinstance(user_config, str) \
        else ExploreConfig.parse_obj(load_config(user_config))
    
    output_dir = Path("explore.ssw_fp") if whether_to_label else Path("explore.ssw")
    process_name = "LASP-ASAP-FP" if whether_to_label else "LASP-ASAP"
    task_list = [exploration, screening_by_e, screening_by_asap, labeling, gen_test_data, testing] if whether_to_label \
        else [exploration, screening_by_e, screening_by_asap]

    if not output_dir.is_dir():
        Path.mkdir(output_dir)

    recordlog = RecordLog(dirpath=output_dir.parent, record="explore.record")
    iter_idx, task_idx = recordlog.check_checkpoint_file()
    task_idx += 1

    explore = wf(config={
        "task_idx": task_idx,
        "iter_idx": iter_idx,
        "task_list": task_list,
        "max_iter": 1
    })

    while iter_idx == 0:
    
        if task_idx==0:
            wflog.info("-" * 75)
            wflog.info(time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime()))
            wflog.info(f"Structure Exploration: {process_name}")
            wflog.info("-" * 75)
            wflog.info("Start Exploring Structures and Screen:")

        explore.run_workflow(
            record=recordlog, 
            user_config=user_config,  
            output_dir=output_dir)

        iter_idx, task_idx = explore.get_idxes
    
        wflog.info("Finish Exploring Structures and Screen!")
        wflog.info('-' * 75)

