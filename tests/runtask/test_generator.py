import copy
from shutil import rmtree
from pathlib import Path
from ase.io import read
from checmate.runtask.generator.vasp import VaspTaskGeneration, vasp_task_factory
from checmate.runtask.generator.cp2k import Cp2kTaskGeneration, cp2k_task_factory
from checmate.runtask.generator.dp import DPTrainTaskGeneration, DPTestTaskGeneration, dptrain_task_factory, dptest_task_factory
from checmate.runtask.generator.lasp import LaspTaskGeneration, lasp_task_factory
from checmate.runtask.generator.lammps import LammpsTaskGeneration, lammps_task_factory



structure = read("../beta-Ga2O3.cif")
setting = {
    "dpdispatcher":{
        "task":{
            "command": "test",
            "forward_files": ["test"],
            "backward_files": ["test"]
        }
    },
}


def remove(dpath="temp_dir"):
    dpath = Path(dpath)
    if dpath.exists():
        rmtree(dpath)


def check_task(task, other_files:list=[]):

    assert task["command"].strip().split(" ")[0] == "test"
    assert sorted(task["forward_files"]) == sorted(["test"]+other_files)
    assert sorted(task["backward_files"]) == sorted(["test"])


def check_dataset(dataset):
        
    assert Path(dataset).is_symlink()



class TestVasp:

    def test_task_generation(self):

        remove() 
        task = VaspTaskGeneration(
            vasp_config=copy.deepcopy(setting)
        ).get_task(task_dir="temp_dir", structure=structure)

        check_task(task)

        files_list = [i.name for i in Path("temp_dir").glob("*")]
        assert sorted(["INCAR", "POSCAR", "POTCAR", "KPOINTS"]) == sorted(files_list)

    
    def test_task_factory1(self):

        remove()
        temp_dir = Path("temp_dir")
        tasks = vasp_task_factory(
            vasp_configs = [copy.deepcopy(setting)]*2,
            structures = [structure],
            output_dir = "temp_dir"
        )

        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("task.000001"), temp_dir.joinpath("task.000002")]) == sorted(tasks.task_dirs)
        files_list = [i.name for i in Path("temp_dir/task.000002").glob("*")]
        assert sorted(["INCAR", "POSCAR", "POTCAR", "KPOINTS"]) == sorted(files_list)


    def test_task_factory2(self):

        remove()
        temp_dir = Path("temp_dir")
        tasks = vasp_task_factory(
            vasp_configs = [copy.deepcopy(setting)],
            structures = [structure]*2,
            output_dir = "temp_dir"
        )

        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("task.000001"), temp_dir.joinpath("task.000002")]) == sorted(tasks.task_dirs) 
        files_list = [i.name for i in Path("temp_dir/task.000002").glob("*")]
        assert sorted(["INCAR", "POSCAR", "POTCAR", "KPOINTS"]) == sorted(files_list)

    
    def test_task_factory3(self):

        remove()
        temp_dir = Path("temp_dir")
        tasks = vasp_task_factory(
            vasp_configs = [copy.deepcopy(setting)]*2,
            structures = [structure]*2,
            output_dir = "temp_dir"
        )
        
        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("task.000001"), temp_dir.joinpath("task.000002")]) == sorted(tasks.task_dirs) 
        files_list = [i.name for i in Path("temp_dir/task.000002").glob("*")]
        assert sorted(["INCAR", "POSCAR", "POTCAR", "KPOINTS"]) == sorted(files_list)

    
    def test_vasp_pre_calc(self):

        remove()
        temp_setting = copy.deepcopy(setting)
        temp_setting["params"] = { "files_to_transfer": {"CONTCAR": None}}

        task = VaspTaskGeneration(
            vasp_config=temp_setting
        ).get_task_from_pre(task_dir="temp_dir", fpath=".")
        
        check_task(task, other_files=["CHGCAR", "CONTCAR"])

        files_list = [i.name for i in Path("temp_dir").glob("*")]
        assert sorted(files_list) == sorted(["INCAR", "POSCAR", "POTCAR", "KPOINTS", "CONTCAR", "CHGCAR"])



class TestCp2k:

    def test_task_generation(self):

        remove()
        task = Cp2kTaskGeneration(
            cp2k_config=copy.deepcopy(setting)
        ).get_task(task_dir="temp_dir", structure=structure)

        check_task(task)

        files_list = [i.name for i in Path("temp_dir").glob("*")]
        assert sorted(["coord.xyz", "cp2k.inp"]) == sorted(files_list)

    
    def test_task_factory(self):

        remove()
        temp_dir = Path("temp_dir")
        tasks = cp2k_task_factory(
            cp2k_configs = [copy.deepcopy(setting)],
            structures = [structure]*2,
            output_dir = "temp_dir" 
        )

        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("task.000001"), temp_dir.joinpath("task.000002")]) == sorted(tasks.task_dirs) 
        files_list = [i.name for i in Path("temp_dir/task.000002").glob("*")]
        assert sorted(["coord.xyz", "cp2k.inp"]) == sorted(files_list)



class TestDP:

    def test_task_generation1(self):

        remove()
        task = DPTrainTaskGeneration(
            dp_config=copy.deepcopy(setting),
            dataset="../dataset",
            type_map=["Ga", "O", "In"]
        ).get_task(task_dir="temp_dir")

        check_task(task)
        check_dataset("dataset")
        Path("dataset").unlink()

        files_list = [i.name for i in Path("temp_dir").glob("*")]
        assert files_list == ["input.json"]


    def test_task_generation2(self):

        remove()
        task = DPTestTaskGeneration(
            dp_config=copy.deepcopy(setting),
            dataset="../dataset"
        ).get_task(task_dir="temp_dir", model_path="../models/frozen_model1.pb")

        check_task(task)
        check_dataset("dataset")
        Path("dataset").unlink()
        Path("temp_dir/frozen_model1.pb").is_symlink()

    
    def test_task_factory1(self):

        remove()
        temp_dir = Path("temp_dir")
        tasks = dptrain_task_factory(
            dp_config=copy.deepcopy(setting),
            numb_train=2,
            dataset="../dataset",
            type_map=["Ga", "O", "In"],
            output_dir="temp_dir"
        )

        check_dataset(temp_dir.joinpath("dataset"))
        temp_dir.joinpath("dataset").unlink()
        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("train.01"), temp_dir.joinpath("train.02")]) == sorted(tasks.task_dirs)
        files_list = [i.name for i in Path("temp_dir/train.02").glob("*")]
        assert files_list == ["input.json"] 

    
    def test_task_factory2(self):
        
        remove()
        temp_dir = Path("temp_dir")
        tasks = dptest_task_factory(
            dp_config=copy.deepcopy(setting),
            models=["../models/frozen_model1.pb", "../models/frozen_model2.pb"],
            dataset="../dataset",
            output_dir="temp_dir"
        )

        check_dataset(temp_dir.joinpath("dataset"))
        temp_dir.joinpath("dataset").unlink()
        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("test.01"), temp_dir.joinpath("test.02")]) == sorted(tasks.task_dirs)
        assert (temp_dir/"test.02"/"frozen_model2.pb").is_symlink()



class TestLasp:

    def test_task_generation(self):
        
        remove()
        temp_setting = copy.deepcopy(setting)
        temp_setting["params"] = {"potential": "lammps"}

        task = LaspTaskGeneration(
            lasp_config=temp_setting,
            potential_config={"params":{"type_map":["Ga", "O", "In"], "models":["../models/frozen_model1.pb"]}}
        ).get_task(task_dir="temp_dir", structure=structure)

        check_task(task, other_files=["in.simple", "data.simple"])

        files_list = [i.name for i in Path("temp_dir").glob("*")]
        assert sorted(["lasp.in", "lasp.str", "in.simple", "data.simple"]) == sorted(files_list)

    
    def test_task_factory(self):

        remove()
        temp_setting = copy.deepcopy(setting)
        temp_setting["params"] = {"potential": "lammps"}

        temp_dir = Path("temp_dir")
        tasks = lasp_task_factory(
            lasp_configs = [copy.deepcopy(temp_setting)],
            potential_configs=[{"params":{"type_map":["Ga", "O", "In"], "models":["../models/frozen_model1.pb"]}}],
            structures = [structure]*2,
            output_dir = "temp_dir"
        )

        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("task.000001"), temp_dir.joinpath("task.000002")]) == sorted(tasks.task_dirs) 
        files_list = [i.name for i in Path("temp_dir/task.000002").glob("*")]
        assert sorted(["lasp.in", "lasp.str", "in.simple", "data.simple"]) == sorted(files_list)
    


class TestLammps:

    def test_task_generation(self):

        remove()
        temp_setting = copy.deepcopy(setting)
        temp_setting["params"] = {"type_map":["Ga", "O", "In"], "models":["../models/frozen_model1.pb"]}
        task = LammpsTaskGeneration(
            lammps_config=temp_setting
        ).get_task(task_dir="temp_dir", structure=structure)

        check_task(task)

        files_list = [i.name for i in Path("temp_dir").glob("*")]
        assert sorted(["in.simple", "data.simple"]) == sorted(files_list) 
    

    def test_task_factory1(self):

        remove()
        temp_setting = copy.deepcopy(setting)
        temp_setting["params"] = {"type_map":["Ga", "O", "In"], "models":["../models/frozen_model1.pb"]}
        temp_dir = Path("temp_dir")
        tasks = lammps_task_factory(
            lammps_configs = [temp_setting],
            structures = [structure]*2,
            output_dir = "temp_dir"
        )

        assert len(tasks.task_list) == 2
        assert sorted([temp_dir.joinpath("task.000001"), temp_dir.joinpath("task.000002")]) == sorted(tasks.task_dirs)
        files_list = [i.name for i in Path("temp_dir/task.000002").glob("*")]
        assert sorted(["in.simple", "data.simple"]) == sorted(files_list)
        remove() 


