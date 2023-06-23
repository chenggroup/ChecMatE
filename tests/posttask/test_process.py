import glob, shutil
import numpy as np
from pathlib import Path

from ase.io import read, write
from dpdata import MultiSystems

from checmate.posttask.process import *


structures = SSWTrajGeneration(
                dirpath="../traj_test"
            ).gen_trajactory()


class TestDatesetGeneration:

    def test_gen_dataset(self, make_temp_dir):

        ms = MultiSystems.from_dir(
            dir_name="../unittask/pre_calc", 
            file_name="OUTCAR", 
            fmt="vasp/outcar", 
            type_map=["O", "In", "Ga"])

        DatasetGeneration(
            dataset="temp_dir"
        ).gen_dataset(ms=ms)

        assert Path("temp_dir").glob("*") != []

        DatasetGeneration(
            dataset="temp_dir"
        ).gen_dataset(ms=ms, iter_idx=0)

        assert Path("temp_dir").glob("system-0-001-*") != []


    def test_screen_dataset_structures(self):

        ms = MultiSystems.from_dir(
            dir_name="../dataset", 
            file_name="In10Ga14O36", 
            fmt="deepmd/npy")

        structs = DatasetGeneration.random_struct(ms=ms, numb_struct_per_system=1)
        assert len(structs) == 1

        structs = DatasetGeneration.random_struct(ms=ms, numb_struct_per_system=2)
        assert len(structs) == 1

    
    def test_move_outliers(self):

        ms = MultiSystems.from_dir(
            dir_name="../dataset", 
            file_name="In10Ga14O36", 
            fmt="deepmd/npy")

        ms, bads = DatasetGeneration.move_outliers(ms=ms, cutoff=0.1)

        assert len(ms)==0 and len(bads)==1



class TestGenTrajactory:

    def test_gen_trajactory(self, make_temp_dir):

        SSWTrajGeneration(
            dirpath="../traj_test",
            whether_to_write=True
        ).gen_trajactory(numb_traj_strs=1289)

        assert Path("temp_dir").glob("traj.xyz") != []


    def test_gen_trajactory2(self, make_temp_dir):

        SSWTrajGeneration(
            dirpath="../traj_test",
            whether_to_write=True
        ).gen_trajactory(numb_traj_strs=1289, ssw_output=True)

        assert Path("temp_dir").glob("traj.xyz") != []



class TestModelDeviProcess:

    def test_model_devi(self, make_temp_dir):
        shutil.copy("../traj_test/model_devi.out", "temp_dir")

        model_force_devi = ModelDeviProcess(
            dirpath="temp_dir",
            bounds=[0.15, 0.3]
        ).get_max_force_devi()

        ModelDeviProcess(
            dirpath="temp_dir",
            bounds=[0.15, 0.3]
        ).get_devi_percent(whether_to_write=True)
        assert Path("temp_dir").glob("percent.json") != []


    def test_gen_candidates(self, make_temp_dir):
        shutil.copy("../traj_test/model_devi.out", "temp_dir")

        ModelDeviProcess(
            dirpath="temp_dir",
            bounds=[0.15, 0.3]
        ).get_candidates(structures, whether_to_write=True)
        assert Path("temp_dir").glob("candidates.xyz") != []



class TestScreenProcess:

    def test_run_filter_by_e(self):

        with open("../traj_test/traj.arc", "r") as f:
            energies = list((round(float(line.strip().split()[-1]),6) for line in f.readlines() if "Energy" in line))

        e_base = min(energies)
        screen_strs = ScreenProcess(structures).run_filter_by_e(energies)

        assert len(screen_strs) == list(np.array(energies)<e_base+7.2).count(True)


    def test_run_filter_by_cluster(self, make_temp_dir):

        shutil.copy("../traj_test/traj.arc", "temp_dir")
        s = read("temp_dir/traj.arc", ":", "dmol-arc")
        write("temp_dir/traj.xyz", s, "extxyz")
        write("temp_dir/one_struct.xyz", s[0], "extxyz")

        ScreenProcess.run_filter_by_cluster(fxyz="temp_dir/traj.xyz", whether_to_write=True, whether_to_plot=True)
        assert Path("temp_dir").glob("cluster_structs.xyz") != []
        assert Path("temp_dir").glob("cluster-pca.png") != []

        cluster_s = ScreenProcess.run_filter_by_cluster(fxyz="temp_dir/one_struct.xyz")
        assert len(cluster_s) == 1

