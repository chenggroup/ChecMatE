import numpy as np
import glob, json, random
from pathlib import Path
from scipy.stats import norm
from functools import partial
from typing import Optional, Union, List

from ase.atoms import Atoms
from ase.io import read, write
from dpdata import MultiSystems, LabeledSystem

from .. import wflog
from ..runtask.parallel import parallel_by_pool
from ..utils.tools.asap import CallASAP
from ..utils.utils import modified_abspath_to_relpath



class DatasetGeneration():

    def __init__(
        self,
        dataset:str,
        bad_data:Optional[str]=None,
        system_prefix:str="system"
    ):
        self.dataset = Path(dataset)
        self.bad_data = bad_data if bad_data is None else Path(bad_data)
        self.system_prefix = system_prefix


    def gen_dataset(self, ms:MultiSystems, iter_idx:Optional[int]=None, cutoff:Union[float, int, None]=None):

        if cutoff:
            ms, bad_ms = self.move_outliers(ms=ms, cutoff=cutoff)
        else:
            bad_ms = MultiSystems()

        if isinstance(iter_idx, int):
            file_prefix = str(self.dataset/self.system_prefix)

            for idx, s in enumerate(ms, start=1):
                s.to_deepmd_npy(f"{file_prefix}-{iter_idx}-{idx:03d}-{s.formula}")
            wflog.info(f"The iteration {iter_idx} adds {ms.get_nframes()} frames to the dataset.")

            if self.bad_data and bad_ms.get_nframes()!=0:
                badfile_prefix = str(self.bad_data/"bad")

                for idx, s in enumerate(bad_ms, start=1):
                    s.to_deepmd_npy(f"{badfile_prefix}-{iter_idx}-{idx:03d}-{s.formula}")
                wflog.info(f"The iteration {iter_idx} adds {bad_ms.get_nframes()} bad frames to the dataset.")

        else:
            ms.to_deepmd_npy(self.dataset)
            
            if self.bad_data and bad_ms.get_nframes()!=0:
                bad_ms.to_deepmd_npy(self.bad_data)
            
            wflog.info(f"The dataset contains {ms.get_nframes()+bad_ms.get_nframes()} frames.")

    
    @staticmethod
    def get_dataset(dirname:str, filename:str, fmt:str="deepmd/npy", type_map:Optional[List[str]]=None) -> MultiSystems:

        dirname = modified_abspath_to_relpath(dirname) if Path(dirname).is_absolute() else Path(dirname)
        
        ms = MultiSystems()
        for i in glob.glob(str(dirname/"**"/filename), recursive=True):
            try:
                ls = LabeledSystem(i, type_map=type_map, fmt=fmt)

                if len(ls) > 0:
                    ms.append(ls)
                else:
                    wflog.info(f"WARNNING: The file {i} can't be added to dataset!")
            except:
                wflog.info(f"WARNNING: The file {i} can't be labelled by dpdata!")
    
        return ms


    @staticmethod
    def get_extra_dataset(dirname:str, common_str:str, fmt:str="deepmd/npy", type_map:Optional[List[str]]=None) -> MultiSystems:

        dirname = modified_abspath_to_relpath(dirname) if Path(dirname).is_absolute() else Path(dirname)

        ms = MultiSystems()
        for i in glob.glob(str(Path(dirname)/"*")):
            if common_str not in i:
                try:
                    ls = LabeledSystem(i, type_map=type_map, fmt=fmt)

                    if len(ls) > 0:
                        ms.append(ls)
                    else:
                        wflog.info(f"WARNNING: The file {i} can't be added to dataset!")
                except:
                    wflog.info(f"WARNNING: The file {i} can't be labelled by dpdata!")
        
        return ms

    @staticmethod
    def move_outliers(ms:MultiSystems, cutoff:Union[float, int]) -> MultiSystems:

        reduce_ms = MultiSystems()
        bad_ms = MultiSystems()
        
        for i in ms:
            for j in i:
                if abs(j.data["forces"]).max() <= cutoff:
                    reduce_ms.append(j)
                else:
                    bad_ms.append(j)
    
        return reduce_ms, bad_ms
        
    @staticmethod
    def random_struct(ms:MultiSystems, numb_struct_per_system:int):

        def screen_fliter(ms:MultiSystems, number:int):
            for s in ms:
                n = s.get_nframes()

                if n >= number:
                    yield s.to_ase_structure()[:number]          
                else:
                    wflog.info(
                        f"The nframe of {s.formula} system is smaller than the given number {number}, so only {n} are chosen.")
                    yield s.to_ase_structure()

        structures = []
        if ms.get_nframes() != 0:
            for chosen_structures in screen_fliter(ms, numb_struct_per_system):
                structures += chosen_structures
        
        return structures

    
class SSWTrajGeneration():

    def __init__(self, dirpath:str, whether_to_write:bool=False):
        
        self.dirpath = Path(dirpath)
        self.whether_to_write = whether_to_write


    def gen_trajactory(self, numb_traj_strs:int|None=None, ssw_output:bool=False):

        traj_strs = self.__collect_traj_structures2() if ssw_output else self.__collect_traj_structures()
        
        if numb_traj_strs and isinstance(numb_traj_strs, int):
            assert len(traj_strs) == numb_traj_strs

        if self.whether_to_write:
            write(str(self.dirpath/"traj.xyz"), traj_strs, format="extxyz") 
        else:
            return traj_strs

    
    def __collect_traj_structures(self):

        dirpath = self.dirpath
        allstrfile = str(dirpath/"allstr.arc")
        minstrfile = str(dirpath/"all.arc")

        all_strs = read(allstrfile, ":", format="dmol-arc")
        min_strs = read(minstrfile, ":", format="dmol-arc")
        with open(allstrfile, "r") as f:
            all_es = list((round(float(line.strip().split()[-1]),6) for line in f.readlines() if "Energy" in line))

        traj_strs = []
        add = traj_strs.append
        remove = min_strs.pop

        for idx, s in enumerate(all_strs):
            if min_strs!=[] and s==min_strs[0]:
                remove(0)

            elif all_es[idx] != -8888.88:
                s.info["ssw_energy"] = all_es[idx]
                add(s)

        assert min_strs == []
        
        return traj_strs
    

    def __collect_traj_structures2(self):

        dirpath = self.dirpath
        allstrfile = str(dirpath/"allstr.arc")
        laspoutfile = str(dirpath/"lasp.out")

        all_strs = read(allstrfile, ":", format="dmol-arc")
        with open(allstrfile, "r") as f:
            all_qs = list((round(float(line[:73].strip().split()[-2]),6) for line in f.readlines() if "Energy" in line))
        with open(laspoutfile, "r") as f:
            traj_qs = list((round(float(line[:73].strip().split()[2]),6) for line in f.readlines() if "Energy,force" in line))
            traj_es = list((round(float(line[:73].strip().split()[0]),6) for line in f.readlines() if "Energy,force" in line))

        traj_strs = []
        add = traj_strs.append
        remove = traj_qs.pop

        for idx, q in enumerate(all_qs):

            if traj_qs != [] and np.isclose(q, traj_qs[0], rtol=0, atol=0.0001):
                all_strs[idx].info["ssw_energy"] = traj_es[len(traj_strs)]
                add(all_strs[idx])
                remove(0)
        
        assert traj_qs == []

        return traj_strs



class ModelDeviProcess():

    def __init__(
        self,  
        bounds:List[float],
        dirpath:str="."
    ):

        self.dirpath = Path(dirpath)
        self.bounds = bounds


    @staticmethod
    def gen_bounds(max_force_devi, hard_criterion:float=0.5):

        mean, sigma = norm.fit(max_force_devi)
        bounds = [mean+2*sigma, mean+3*sigma]
        if mean > hard_criterion:
            bounds = [0.15, 0.3]

        elif bounds[0] > hard_criterion:
            bounds = [0.15, hard_criterion]

        elif bounds[1] > hard_criterion:
            bounds[1] = hard_criterion

        wflog.info("The bounds of this model deviation: [{0:0.2f}, {1:0.2f}]".format(bounds[0], bounds[1])) 
        return bounds


    def get_devi_percent(self, max_force_devi:Optional[np.array]=None, whether_to_write:bool=False, max_devi:Union[float, int]=1):

        max_force_devi = self.get_max_force_devi() if max_force_devi is None else np.array(max_force_devi)
        bounds = self.bounds

        tot = len(max_force_devi)
        assert tot != 0

        accurate_mask = list(max_force_devi <= bounds[0])
        failed_mask = list(max_force_devi > bounds[1])
        candidate_mask = list((bounds[0] < max_force_devi) & ( max_force_devi <= bounds[1]))

        percent = {
            "fail": round(failed_mask.count(True) / tot, 6),
            "candidate":  round(candidate_mask.count(True) / tot, 6),
            "accuracy": round(accurate_mask.count(True) / tot, 6),
            "max_force_devi": round(max_force_devi.max(), 6),
            "mean": round(norm.fit(max_force_devi[max_force_devi<=max_devi])[0], 6)
        }

        if whether_to_write and not("*" in str(self.dirpath)):
            with open(str(self.dirpath/"percent.json"), "w") as f:
                json.dump(percent, f, indent=4)
        
        else:
            return percent

    
    def get_max_force_devi(self, column:int=4):
        
        devifile = str(self.dirpath/"model_devi.out")

        if "*" in devifile:
            max_force_devi = []
            devifiles = glob.glob(devifile)
            assert devifiles != []
            for i in list((np.loadtxt(f)[:, column] for f in devifiles)):
                max_force_devi.extend(i)

        else:
            assert Path(devifile).is_file()
            max_force_devi = np.loadtxt(devifile)[:, column]
        
        return np.array(max_force_devi)


    def get_candidates(self, traj_strs:List[Atoms], max_force_devi=None, whether_to_write:bool=False):

        max_force_devi = self.get_max_force_devi() if max_force_devi is None else max_force_devi
        bounds = self.bounds
        assert len(traj_strs) == max_force_devi.size
        
        full_ids = np.arange(max_force_devi.size)
        candidate_mask = (bounds[0] <= max_force_devi) & ( max_force_devi < bounds[1])
        candidates = list((traj_strs[i] for i in full_ids[candidate_mask]))

        if whether_to_write and not("*" in str(self.dirpath)):
            write(str(self.dirpath/"candidates.xyz"), candidates, format="extxyz")
        else:
            return candidates

    
    def get_fails(self, traj_strs:List[Atoms], max_force_devi=None, whether_to_write:bool=False):

        max_force_devi = self.get_max_force_devi() if max_force_devi is None else max_force_devi
        bounds = self.bounds
        assert len(traj_strs) == max_force_devi.size

        full_ids = np.arange(max_force_devi.size)
        fail_mask = (max_force_devi >= bounds[1])
        fails = list((traj_strs[i] for i in full_ids[fail_mask]))

        if whether_to_write and not("*" in self.dirpath):
            write(str(self.dirpath/"fails.xyz"), fails, format="extxyz")
        else:
            return fails



class ScreenProcess():

    def __init__(self, structures:Optional[list[Atoms]]=None, filenames:Union[List[str], str, None]=None, fmt:Optional[str]=None, output_dir:Optional[str]=None):

        assert not(structures is None and filenames is None)

        if structures is None:
            filenames = filenames if isinstance(filenames, list) else [filenames]
            s = []
            for f in filenames:
                s.extend(read(f, ":", format=fmt))
            self.structures = s

        else:
            self.structures = structures

        self.output_dir=Path(output_dir) if output_dir is not None else None


    def run_filter_by_e(self, energies:Optional[List[float]]=None, e_cutoff:Union[int,float]=0.1):

        structures = self.structures

        if energies:
            assert len(structures) == len(energies)
            energies = list((i[0]/i[1].get_global_number_of_atoms() for i in zip(energies, structures)))
            e_base = min(energies)
            screen_strs = list((structures[idx] for idx, e in enumerate(energies) if e-e_base < e_cutoff))

        else:
            energies = list((i.get_total_energy()/i.get_global_number_of_atoms() for i in structures))
            e_base = min(energies)
            screen_strs = list((structures[idx] for idx, e in enumerate(energies) if e-e_base < e_cutoff))

        if self.output_dir:
            write(str(self.output_dir/"screen_structs_by_e.xyz"), screen_strs, format="extxyz")

        else:
            return  screen_strs


    @staticmethod
    def run_filter_by_cluster(
        fxyz,
        e_sort:bool=False,
        asap_config:Optional[dict]=None, 
        numb_struct_per_label:int=1, 
        noise_percent: Union[float,int]=100, 
        whether_to_write:bool=False, 
        whether_to_plot:bool=False,
        **kwargs):
        
        assert isinstance(noise_percent, (float, int)) and noise_percent>=0
        assert Path(fxyz).is_file()
        
        output_dir = Path(fxyz).parent
        callasap = CallASAP(
            fxyz=fxyz,
            user_setting=asap_config, 
            output_dir=output_dir,
            **kwargs
        )
        
        structs = callasap.asapxyz.frames
        cluster_structures = []
        noise = []

        if len(structs) > 50:
            try:
                label_ids = callasap.cluster(whether_to_plot=whether_to_plot)
                for i in label_ids:
                    if i!=-1:
                        if e_sort:
                            label_ids[i].sort(key=lambda s:s.info["energy"])
                        else:
                            random.shuffle(label_ids[i])
                        if len(label_ids[i]) > numb_struct_per_label:
                            cluster_structures += list((structs[j] for j in label_ids[i][:numb_struct_per_label]))
                        else:
                            cluster_structures += list((structs[j] for j in label_ids[i]))
                    else:
                        random.shuffle(label_ids[-1])
                        noise.extend([structs[i] for i in label_ids[-1][:int(len(label_ids[-1])*noise_percent/100)]])             

            except Exception as e:
                wflog.info(f"{output_dir}: {e.args}")
                random.shuffle(structs)
                cluster_structures = structs

        else:
            wflog.info(f"The number of structure in {output_dir} is less than 50.")
            random.shuffle(structs)
            cluster_structures = structs

        if whether_to_write:
            write(str(output_dir/"cluster_structs.xyz"), cluster_structures+noise, format='extxyz')
        else:
            return cluster_structures+noise



class CandidateProcess():

    def __init__(
        self,
        dirpaths:Union[List[str], str],
        output_file:str,
        bounds:List[float],
        n_process: int=4,
        ssw_output: bool=False
    ):

        assert isinstance(bounds, list)
        self.bounds = bounds
        self.output_file = output_file
        self.n_process = n_process
        self.dirpaths = dirpaths if isinstance(dirpaths, list) else glob.glob(dirpaths)
        self.ssw_output = ssw_output


    def gen_candidates_ensemble(self, whether_to_return:bool=True):

        func = partial(self.candidates_indicator, whether_to_write=not(whether_to_return))

        if whether_to_return:
            results = parallel_by_pool(func, self.dirpaths, whether_to_return=whether_to_return, n_process=self.n_process)
            candidates_ensemble = list((result for result in results if result is not None))

            return candidates_ensemble
        
        else:
            parallel_by_pool(func, self.dirpaths)


    def candidates_indicator(self, dirpath:str, whether_to_write:bool=False, Traj=SSWTrajGeneration, Devi=ModelDeviProcess):

        bounds = self.bounds

        traj_strs = Traj(dirpath=dirpath).gen_trajactory(ssw_output=self.ssw_output)
        candidates = Devi(dirpath=dirpath, bounds=bounds).get_candidates(traj_strs, whether_to_write=whether_to_write)
        
        if not(whether_to_write):
            return candidates
    

    def filing_chosen_candidates(self, numb_candidate_per_traj:int=10, whether_to_cluster:bool=True, Filter=ScreenProcess, **kwargs):

        if whether_to_cluster:
            self.gen_candidates_ensemble(whether_to_return=not(whether_to_cluster))
            func = partial(Filter.run_filter_by_cluster, **kwargs)
            files = list((str(Path(dirpath).joinpath("candidates.xyz")) for dirpath in self.dirpaths))
            results = list(parallel_by_pool(func, files, whether_to_return=True))

        else:
            candidates_ensemble = self.gen_candidates_ensemble()
            results = candidates_ensemble

        filing_chosen_candidates = []

        for i in results:
            if len(i) <= numb_candidate_per_traj:
                filing_chosen_candidates.extend(i)
            
            else:
                random.shuffle(i)
                filing_chosen_candidates.extend(i[:numb_candidate_per_traj])
        
        wflog.info(f"The total number of candidates is {len(filing_chosen_candidates)}.")
        write(self.output_file, filing_chosen_candidates, format="extxyz")



class DPProcess():

    @staticmethod
    def dptest_file_process(filename:str, dataset:Optional[str]=None, peratom:bool=False, force_split:bool=False):
        
        data = np.loadtxt(filename)

        if peratom or force_split:
            assert Path(dataset).is_dir()

            with open(filename, "r") as f:
                system_dirs = list((Path(dataset)/Path(i.strip("# ").split(":")[0]).name for i in f.readlines() if "#" in i))

            nframe = list((LabeledSystem(system_dir, fmt="deepmd/npy").get_nframes() for system_dir in system_dirs))
            natom = list((len(open(str(Path(system_dir)/"type.raw")).readlines()) for system_dir in system_dirs))

            natoms = []
            for i, j in zip(nframe, natom):
                natoms += list((j for _ in range(i)))

            if peratom:

                return data/np.array(list(([i]*2) for i in natoms))

            elif force_split:
                idx = list((sum(natoms[:i]) for i in range(1,len(natoms))))
                assert idx[-1]+idx[-2] == len(data)-1

                return np.split(data, idx)

        else:

            return data



class DpdataFilter():

    def __init__(self, dirname:str, filename:str, fmt:str="deepmd/npy", filter_type:str="force", force_upper:float|None=None, energy_upper:float|None=None):

        self.ms = MultiSystems.from_dir(dirname, filename, fmt=fmt)
        self.filter_type = filter_type
        self.force_upper = force_upper
        self.energy_upper = energy_upper
        

    def force_filter(self, ms:MultiSystems|None=None):

        ms = self.ms if ms is None else ms

        modified_ms = MultiSystems()
        add = modified_ms.append

        assert self.force_upper is not None
        for s in (j for i in ms for j in i if abs(j.data["forces"][0]).max() < self.force_upper):
            add (s)
        
        return modified_ms
    

    def energy_filter(self, ms:MultiSystems|None=None, MS=MultiSystems):

        ms = self.ms if ms is None else ms

        modified_ms = MultiSystems()
        add = modified_ms.append

        assert self.energy_upper is not None
        for s in (j for i in ms for j in i if j.data["energies"][0]/j.get_natoms() < self.energy_upper):
            add (s)
        
        return modified_ms
    

    def run_filter(self) -> MultiSystems:

        if self.filter_type == "force":
            modified_ms = self.force_filter()
    
        elif self.filter_type == "energy":
            modified_ms = self.energy_filter()

        elif self.filter_type == "both":
            ms1 = self.force_filter()
            modified_ms = self.energy_filter(ms=ms1)

        else:
            raise TypeError(f"The filter_type {self.filter_type} is wrong!")

        return modified_ms
    

    def write_output(self, output_dir:str="./", file_prefix:str|None=None):

        for system in self.run_filter():

            if file_prefix:
                file_name = "-".join([file_prefix, system.formula])
                system.to_deepmd_npy(str(Path(output_dir)/file_name))
            
            else:
                system.to_deepmd_npy(str(Path(output_dir)/system.formula))

