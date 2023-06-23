import json, glob, copy
from pathlib import Path
from os.path import abspath
from typing import Union, List, Optional

from dpdispatcher.machine import Machine
from dpdispatcher.submission import Submission, Resources, Task

from .generator import DpdispatcherConfig
from ..pretask.inputs import RunInput



class RunTask():
    """
    Run Task by DP-Dispatcher

    Parameters
    ----------
    output_dir: 
        directory where the task is running
    dpdispatcher_config: 
        setting about dpdispatcher
    """

    def __init__(self,
                 output_dir: Union[str, Path],
                 dpdispatcher_config: Union[DpdispatcherConfig, dict],
                 make_dir_if_not_present: bool=True
        ):

        output_dir = Path(output_dir)
        if make_dir_if_not_present and not output_dir.exists():
            Path.mkdir(output_dir)

        dpdispatcher_config = copy.deepcopy(DpdispatcherConfig.parse_obj(dpdispatcher_config))

        machine, resources = RunInput(
            machine = dpdispatcher_config.machine,
            resources = dpdispatcher_config.resources,
            resources_temp_key = dpdispatcher_config.resources_temp_key
        ).machine_and_resources

        dpdispatcher_config.machine = machine
        dpdispatcher_config.resources = resources
        with open(str(output_dir/"machine.json"), "wt") as f:
            json.dump({"machine":machine, "resouces":resources}, f, indent=2)

        self.output_dir = output_dir
        self.dpdispatcher_config = dpdispatcher_config


    def link_forward_common_files(self) -> None:
        
        forward_common_files = self.dpdispatcher_config.forward_common_files 
        common_files = []
        for i in forward_common_files:
            files = glob.glob(abspath(i))
            print(i)
            assert files != []
            common_files += files
        
        output_dir = self.output_dir
        temp_forward_common_files = []

        for i in common_files:
            fpath = Path(i)
            temp_forward_common_files.append(fpath.name)
            des = output_dir/fpath.name

            if not(des.exists()):
                des.symlink_to(fpath)
        
        self.dpdispatcher_config.forward_common_files = temp_forward_common_files


    def check_forward_common_files(self) -> None:

        output_dir=self.output_dir
        for i in self.dpdispatcher_config.forward_common_files:
            (output_dir/i).is_file()


    def submit_task(self, task_list:List[Task], whether_to_run:bool=True) -> Optional[Submission]:

        dpdispatcher_config = self.dpdispatcher_config
        if dpdispatcher_config.forward_common_files:
            self.link_forward_common_files()
            self.check_forward_common_files()

        assert task_list != []
        
        
        submission = Submission(
            work_base=str(self.output_dir),
            machine=Machine.load_from_dict(dpdispatcher_config.machine),
            resources=Resources.load_from_dict(dpdispatcher_config.resources),
            task_list=task_list,
            forward_common_files=dpdispatcher_config.forward_common_files,
            backward_common_files=dpdispatcher_config.backward_common_files
        )

        if whether_to_run:
            submission.run_submission()
        else:
            return submission

