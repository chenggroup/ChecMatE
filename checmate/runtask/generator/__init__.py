from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel
from dataclasses import dataclass
from dpdispatcher.submission import Task



class DpdispatcherConfig(BaseModel):
    machine: Optional[dict] = None
    resources: Optional[dict] = None
    task: dict = {}
    resources_temp_key:str="general"
    forward_common_files: List[str] = []
    backward_common_files: List[str] = []


class GeneralUserConfig(BaseModel):
    template_path: Optional[str] = None
    params: dict = {}
    dpdispatcher: DpdispatcherConfig = DpdispatcherConfig.parse_obj({})


class BaseTaskGeneration(ABC):

    user_config: GeneralUserConfig
 
    @abstractmethod
    def gen_task_inputs(self, task_dir:str, **kwargs) -> None:
        pass

    @abstractmethod
    def get_task(task_dir:str, **kwargs) -> Task:
        pass


@dataclass
class GeneralFactaryOutput():
    task_dirs: List[str]
    task_list: List[Task]