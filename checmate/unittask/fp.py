import copy
from re import compile
from pathlib import Path
from typing import Union, List, Optional
from monty.re import regrep

from pydantic import BaseModel
from pymatgen.io.vasp.outputs import Vasprun

from .. import wflog
from ..runtask.run import RunTask
from ..runtask.generator import GeneralUserConfig, GeneralFactaryOutput
from ..runtask.generator.vasp import vasp_task_factory, vasp_task_factory_from_pre
from ..runtask.generator.cp2k import cp2k_task_factory
from ..runtask.parallel import parallel_by_pool


def check_vasp_input(fpath:Union[str, Path], whether_from_pre:bool=False):

    incars = Path(fpath).glob("**/INCAR")
    assert incars != []
    if whether_from_pre:
        chgcars = Path(fpath).glob("**/CHGCAR")
        assert chgcars != []


def check_cp2k_input(fpath:Union[str, Path]):

    inps = Path(fpath).glob("**/cp2k.inp")
    assert inps != []


def check_vasp_converge(fpath:Union[str, Path]):

    try:
        vpr = Vasprun(fpath)
        if not vpr.converged:
            wflog.info(f"Warning: The task {Path(fpath).parent} is not converged!")
    except Exception as e:
        wflog.info(f"The task {Path(fpath).parent}: {e}")



def check_cp2k_converge(fpath:Union[str, Path]):

    patterns1 = {"completed": compile(r"PROGRAM ENDED AT\s+(\w+)")}
    patterns2 = {"scf_converged": compile(r"SCF run converged\s+(\w+)")}

    completed = regrep(str(fpath), patterns1, reverse=True, terminate_on_match=True, postprocess=bool).get("completed")
    scf_converged = regrep(str(fpath), patterns2, reverse=True, terminate_on_match=True, postprocess=bool).get("scf_converged") 

    if completed:
        if not scf_converged:
            wflog.info(f"Warning: There is at least one unconverged SCF cycle in the task {Path(fpath).parent}.")
    else:
        wflog.info(f"Warning: The task {Path(fpath).parent} did not finish!")



def vasp_flow(
    user_config:Union[dict, GeneralUserConfig],
    structures: list,
    output_dir:Union[str, Path]=".", 
    whether_to_cover:bool=False, 
    whether_to_check:bool=True,
    whether_to_return:bool=False):

    user_config = copy.deepcopy(GeneralUserConfig.parse_obj(user_config))

    tasks = vasp_task_factory(
        vasp_configs=[user_config],
        structures=structures,
        output_dir=output_dir,
        whether_to_cover=whether_to_cover)
    
    runner = RunTask(
        output_dir=output_dir,
        dpdispatcher_config=user_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)

    check_vasp_input(fpath=output_dir)
    
    runner.run_submission()
    
    if whether_to_check:
        files = list((Path(i).joinpath("vasprun.xml") for i in tasks.task_dirs))
        parallel_by_pool(check_vasp_converge, files)
    
    if whether_to_return:
        return tasks



def vasp_nscf_flow(
    user_config:Union[dict, GeneralUserConfig],
    fpaths:Union[List[str], List[Path]],
    output_dir:Union[str, Path]=".", 
    whether_to_cover:bool=False, 
    whether_to_check:bool=True,
    whether_to_return:bool=False,
    **kwargs):

    user_config = copy.deepcopy(GeneralUserConfig.parse_obj(user_config))

    tasks = vasp_task_factory_from_pre(
        vasp_configs=[user_config], 
        fpaths=fpaths,
        output_dir=output_dir, 
        whether_to_cover=whether_to_cover, 
        **kwargs)
    
    runner = RunTask(
            output_dir=output_dir,
            dpdispatcher_config=user_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)

    check_vasp_input(fpath=output_dir, whether_from_pre=True)

    runner.run_submission()

    if whether_to_check:
        files = list((Path(i).joinpath("vasprun.xml") for i in tasks.task_dirs)) 
        parallel_by_pool(check_vasp_converge, files)

    if whether_to_return:
        return tasks



def cp2k_flow(
    user_config:Union[dict, GeneralUserConfig],
    structures: list,
    output_dir:Union[str, Path]=".", 
    whether_to_cover:bool=False, 
    whether_to_check:bool=True,
    whether_to_return:bool=False):

    user_config = copy.deepcopy(GeneralUserConfig.parse_obj(user_config))

    tasks = cp2k_task_factory(
        cp2k_configs=[user_config], 
        structures=structures,
        output_dir=output_dir, 
        whether_to_cover=whether_to_cover)

    runner = RunTask(
            output_dir=output_dir,
            dpdispatcher_config=user_config.dpdispatcher
        ).submit_task(task_list=tasks.task_list, whether_to_run=False)

    check_cp2k_input(fpath=output_dir)

    runner.run_submission()

    if whether_to_check:
        files = list((Path(i).joinpath("cp2k.out") for i in tasks.task_dirs)) 
        parallel_by_pool(check_cp2k_converge, files)

    if whether_to_return:
        return tasks



class VaspTestConfig(BaseModel):
    encuts: Optional[List[int]]=None
    k_r_densities: Optional[List[int]]=None
    vasp_config: GeneralUserConfig


def vasp_test(
        user_config:Union[dict, VaspTestConfig],
        structure,
        output_dir:Union[str, Path]=".", 
        whether_to_cover:bool=False, 
        whether_to_check:bool=True,
        whether_to_return:bool=False):

    from ..pretask.sets import update_dict
    from ..runtask.generator.vasp import VaspTaskGeneration

    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    user_config = copy.deepcopy(VaspTestConfig.parse_obj(user_config))
    
    task_list = []
    task_dirs = []
    encuts, k_r_densities = user_config.encuts, user_config.k_r_densities
    assert not(encuts is None and k_r_densities is None)

    if isinstance(encuts, list):
        temp_setting = copy.deepcopy(user_config.vasp_config.dict())
        
        for encut in encuts:
            task_dir = output_dir.joinpath(f"encut_{encut}")
            update_dict(temp_setting, {"params":{"incar":{"ENCUT":encut}}})

            task_list.append(VaspTaskGeneration(
                vasp_config=temp_setting
            ).get_task(task_dir=task_dir, structure=structure))
            task_dirs.append(task_dir)
            
    if isinstance(k_r_densities, list):
        temp_setting = copy.deepcopy(user_config.vasp_config.dict())

        for k_density in k_r_densities:
            task_dir = output_dir.joinpath(f"kpoints_{k_density}")
            update_dict(temp_setting, {"params":{"kpoints":{"reciprocal_density":k_density}}})

            task_list.append(VaspTaskGeneration(
                vasp_config=temp_setting,
                whether_to_cover=whether_to_cover
            ).get_task(task_dir=task_dir, structure=structure))
            task_dirs.append(task_dir)

    runner = RunTask(
        output_dir=output_dir, 
        dpdispatcher_config=user_config.vasp_config.dpdispatcher
    ).submit_task(task_list=task_list, whether_to_run=False)

    check_vasp_input(fpath=output_dir)

    runner.run_submission()

    if whether_to_check:
        files = list((Path(i).joinpath("vasprun.xml") for i in task_dirs))
        parallel_by_pool(check_vasp_converge, files)
    
    if whether_to_return:
        return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)


class Cp2kTestConfig(BaseModel):
    cutoffs: List[int]
    cp2k_config: GeneralUserConfig


def cp2k_test(
        user_config:Union[dict, Cp2kTestConfig],
        structure,
        output_dir:Union[str, Path]=".", 
        whether_to_cover:bool=False, 
        whether_to_check:bool=True,
        whether_to_return:bool=False):
 
    from ..pretask.sets import update_dict
    from ..runtask.generator.cp2k import Cp2kTaskGeneration
    
    output_dir = Path(output_dir)
    if not output_dir.exists():
        Path.mkdir(output_dir)

    user_config = copy.deepcopy(Cp2kTestConfig.parse_obj(user_config))

    task_list = [] 
    task_dirs = []
    cutoffs = user_config.cutoffs
    temp_setting = copy.deepcopy(user_config.cp2k_config.dict())

    for cutoff in cutoffs:
        task_dir = output_dir.joinpath(f"cutoff_{cutoff}")
        update_dict(temp_setting, {"params":{"FORCE_EVAL": {"DFT": {"MGRID": {"CUTOFF": cutoff}}}}})

        task_list.append(Cp2kTaskGeneration(
            cp2k_config=temp_setting,
            whether_to_cover=whether_to_cover
        ).get_task(task_dir=task_dir, structure=structure))
        task_dirs.append(task_dir)

    runner = RunTask(
        output_dir=output_dir, 
        dpdispatcher_config=user_config.cp2k_config.dpdispatcher
    ).submit_task(task_list=task_list, whether_to_run=False) 

    check_cp2k_input(fpath=output_dir)

    runner.run_submission()

    if whether_to_check:
        files = list((Path(i).joinpath("cp2k.out") for i in task_dirs)) 
        parallel_by_pool(check_cp2k_converge, files)
    
    if whether_to_return:
        return GeneralFactaryOutput(task_dirs=task_dirs, task_list=task_list)
