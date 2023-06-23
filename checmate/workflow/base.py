import glob, time
from pathlib import Path
from random import shuffle
from typing import Optional, Union

from ase.io import read
from pydantic import BaseModel

from .. import wflog
from ..posttask.process import ScreenProcess


class WorkflowGeneralConfig(BaseModel):
    task_idx: int
    iter_idx: int
    task_list: list
    max_iter: int = 30

    

class WorkflowGeneration():

    cont = True

    def __init__(self, config: Union[WorkflowGeneralConfig, dict]):

        self.config = WorkflowGeneralConfig.parse_obj(config)


    def run_workflow(self, record, **kwargs):

        self.check_task_idx
        self.whether_stop_workflow()
        
        task_list = self.config.task_list
        numb_task = len(task_list)

        while self.config.task_idx < numb_task and self.cont:

            task = task_list[self.config.task_idx]
            record.log_task(
                getattr(task, "__name__"), self.config.task_idx, self.config.iter_idx)

            task(**kwargs)

            record.record_task(self.config.iter_idx, self.config.task_idx)
            self.config.task_idx += 1
            
    

    def whether_stop_workflow(self):

        if self.config.iter_idx >= self.config.max_iter:
            self.set_cont(cont=False)
            wflog.info("-" * 60)
            wflog.info("This workflow reaches the max iteration number and is stopped.")
            wflog.info("-" * 60)


    def add_task(self, task, task_idx:Optional[int]=None):

        if task not in self.config.task_list:
            if task_idx is None:
                self.config.task_list.append(task)
            else:
                self.config.task_list.insert(task_idx, task)
        else:
            raise NameError("the name of this task has already been in this class !")
    

    @classmethod
    def get_cont(cls):

        return cls.cont


    @classmethod
    def set_cont(cls, cont:bool):

        cls.cont = cont


    @property
    def get_idxes(self):

        self.check_task_idx

        return self.config.iter_idx, self.config.task_idx


    @property
    def check_task_idx(self):

        task_idx = self.config.task_idx
        numb_task = len(self.config.task_list)
        if task_idx == numb_task:
            self.config.iter_idx += 1
            self.config.task_idx = 0

        elif task_idx not in range(numb_task):
            raise RuntimeError ("unknown task %d, something wrong" % task_idx)



class RecordLog():

    def __init__(self, record:Union[str, Path]="gen_mlp.record", dirpath:str="."):

        self.record = Path(dirpath).joinpath(record)


    def check_checkpoint_file(self):

        rf = self.record
        idxes_rec = [0, -1]

        if rf.is_file() and rf.stat().st_size!=0:

            with open(str(rf), 'r') as frec:
                lines = frec.readlines()
                idxes_rec = [int(x) for x in lines[-1].split()]

            wflog.info("continue from iter %03d task %03d" % (
                idxes_rec[0], idxes_rec[1]))
        
        return idxes_rec[0], idxes_rec[1]


    def record_task(self, iter_idx:int, task_idx:int):

        with open(str(self.record), "a") as frec:
            frec.write("{0:d} {1:d}\n".format(iter_idx, task_idx))

    
    @staticmethod
    def log_task(taskname:str, task_idx:int, iter_idx:int|None=None):

        if iter_idx is None:
            wflog.info((time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime())+"  task {1:03d}: {2:s}").format(
            iter_idx, task_idx, taskname))
        
        else:
            wflog.info((time.strftime("[ %Y-%m-%d %H:%M:%S ]", time.localtime())+"  iter {0:03d} task {1:03d}: {2:s}").format(
                iter_idx, task_idx, taskname))

    
    def cache_key(self):

        return self.get_iter_idx()


    def get_iter_idx(self):

        with open(self.record, "r") as frec:
            lines = frec.readlines()
            idxes_rec = [int(x) for x in lines[-1].split()]

        return idxes_rec[0]



class InitialStructConfig(BaseModel):
    structure_paths: Union[str, list]
    structure_format: str
    numb_struct_per_file: int = 0


class InitialStruct():

    def __init__(self, config:Union[dict, InitialStructConfig]):

        self.config = InitialStructConfig.parse_obj(config)

    
    def collect_structures(self, whether_to_cluster:bool=False, **kwargs):

        structure_paths = self.config.structure_paths
        structure_paths = sorted(structure_paths) if isinstance(structure_paths, list) else sorted(glob.glob(structure_paths))
        assert structure_paths != []
 
        results = list((self.__get_structures(s_path=str(p), whether_to_cluster=whether_to_cluster, **kwargs) for p in structure_paths))
        
        structure_list = []
        extend = structure_list.extend
        for i in results:
            extend(i)

        return structure_list


    def __get_structures(self, s_path:str, whether_to_cluster:bool=False, **kwargs):

        fmt = self.config.structure_format
        numb_structures = self.config.numb_struct_per_file

        if whether_to_cluster:
            structures = ScreenProcess.run_filter_by_cluster(
                fxyz=s_path, fileformat="{'format':'%s'}" % fmt, **kwargs)
        else:
            structures = read(s_path, index=":", format=fmt)

        if numb_structures:
            shuffle(structures)
            return structures[:numb_structures] \
                if len(structures) > numb_structures else structures
        else:
            return structures


