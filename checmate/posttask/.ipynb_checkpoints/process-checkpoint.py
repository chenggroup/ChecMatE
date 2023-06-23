import glob, json, random
import numpy as np
from os import path
from functools import partial

from ase.atoms import Atoms
from ase.io import read, write
from dpdata import MultiSystems, LabeledSystem

from .. import wflog
from ..runtask.parallel import parallel_by_pool, parallel_by_threadpool
from ..utils.utils import dp_type_map, f_remove, get_mean_and_sigma
from ..utils.tools.asap import CallASAP



class DpdataSetGeneration():

    def __init__(
        self,
        dirname:str,
        filename:str,
        fmt:str="auto", 
        type_map:list[str]|None=None,
        cutoff:int|float|None=None   #the cutoff of the maximum atomic force
    ):
        self.dirname = dirname
        self.filename = filename
        self.fmt = fmt
        self.type_map = type_map
        self.cutoff = cutoff


    def gen_dataset(self, dataset:str, iter_numb:int|None=None, bad_data:str|None=None):

        systems = glob.glob(path.join(dataset, f"system*.001"))
        if systems!=[] and self.type_map is None:
            self.type_map = dp_type_map(systems[-1])

        ms = self.get_dataset()
        bads = MultiSystems()
        if isinstance(self.cutoff, (int, float)) and self.cutoff:
            ms, bads = self.move_outliers(ms=ms)

        if type(iter_numb)==int:
            
            file_prefix = path.join(dataset, "system")
            for idx, s in enumerate(ms, start=1):
                s.to_deepmd_npy(f"{file_prefix}-{iter_numb}.{idx:03d}")
            wflog.info(f"The iteration {iter_numb} adds {ms.get_nframes()} structures to the dataset.")

            if bads.get_nframes()!=0 and bad_data!=None:
                badfile_prefix = path.join(bad_data, "bad")
                for idx, s in enumerate(bads, start=1):
                    s.to_deepmd_npy(f"{badfile_prefix}-{iter_numb}.{idx:03d}")
                wflog.info(f"The iteration {iter_numb} adds {bads.get_nframes()} bad structures to the dataset.")

        else:
            ms.to_deepmd_npy(dataset)
            
            if bads.get_nframes()!=0 and bad_data!=None:
                bads.to_deepmd_npy(bad_data)

    

    def get_dataset(self):

        dirname = self.dirname
        filename = self.filename
        fmt = self.fmt
        type_map = self.type_map
        
        if filename != "*":
            ms = MultiSystems.from_dir(
                dir_name=dirname, 
                file_name=filename, 
                fmt=fmt, 
                type_map=type_map
            )
        else:
            ms = MultiSystems()
            for i in glob.glob(path.join(dirname, "*")):
                ms.append(
                    MultiSystems.from_dir(
                        dir_name=dirname,
                        file_name=path.basename(i),
                        fmt=fmt,
                        type_map=type_map
                    )
                )

        return ms

    
    def move_outliers(self, ms:MultiSystems|None=None):

        if ms is None:
            ms = self.get_dataset()

        reduce_ms = MultiSystems()
        bad_ms = MultiSystems()
        
        assert self.cutoff is not None
        for i in ms:
            for j in i:
                if abs(j.data["forces"]).max() <= self.cutoff:
                    reduce_ms.append(j)
                else:
                    bad_ms.append(j)
    
        return reduce_ms, bad_ms
        

    def screen_dataset_structures(self, numb_structure_per_system:int=1):

        structures = []

        for chosen_structures in self.__screen_fliter(numb_structure_per_system):
            structures += chosen_structures
        
        return structures


    def __screen_fliter(self, numb_structure:int):

        ms = self.get_dataset()

        for s in ms:

            if s.get_nframes() > numb_structure:
                yield s.to_ase_structure()[:numb_structure]
            
            else:
                wflog.info(
                    f"The nframe of {s.formula} system is smaller than the given number {numb_structure}, so only the {s.get_nframes()} are chosen.")
                yield s.to_ase_structure()


    def check_type_map(self, dirpaths:str|list[str]|None=None, numb_type:int=1):
        
        if dirpaths:
            dirpaths = dirpaths if isinstance(dirpaths, list) else glob.glob(dirpaths)
        else:
            dirpaths = glob.glob(path.join(self.dirname, self.filename))

        assert dirpaths != []
        
        all_type_map = []
        add = all_type_map.append

        for type_map in (dp_type_map(dirpath) for dirpath in dirpaths):
            add(str(type_map))
        assert len(set(all_type_map)) == numb_type
        


class SSWTrajGeneration():

    def __init__(self, dirpath:str, whether_to_write:bool=False, ssw_output:bool=False):
        
        self.dirpath = dirpath
        self.whether_to_write = whether_to_write
        self.ssw_output = ssw_output


    def gen_trajactory(self, numb_traj_strs:int|None=None):

        if self.ssw_output:
            traj_strs = self.__collect_traj_structures1()
        else:
            traj_strs = self.__collect_traj_structures2()
        
        if numb_traj_strs and isinstance(numb_traj_strs, int):
            assert len(traj_strs) == numb_traj_strs

        if self.whether_to_write:
            write(path.join(self.dirpath, "traj.xyz"), traj_strs, format="extxyz") 
        else:
            return traj_strs

            
    def __collect_traj_structures1(self):

        dirpath = self.dirpath
        allstrfile = path.join(dirpath, "allstr.arc")
        laspoutfile = path.join(dirpath, "lasp.out")

        all_strs = read(allstrfile, ":", format="dmol-arc")
        with open(allstrfile, "r") as f:
            all_es = list((round(float(line[:73].strip().split()[-1]),6) for line in f.readlines() if "Energy" in line))
        with open(laspoutfile, "r") as f:
            traj_es = list((round(float(line[:73].strip().split()[1]),6) for line in f.readlines() if "Energy,force" in line))

        traj_strs = []
        add = traj_strs.append
        remove = traj_es.pop

        for idx, e in enumerate(all_es):

            if traj_es != [] and abs(e-traj_es[0]) < 0.000002:
                all_strs[idx].info["ssw_energy"] = e
                add(all_strs[idx])
                remove(0)
        
        assert traj_es == []

        return traj_strs

    
    def __collect_traj_structures2(self):

        dirpath = self.dirpath
        allstrfile = path.join(dirpath, "allstr.arc")
        minstrfile = path.join(dirpath, "all.arc")

        all_strs = read(allstrfile, ":", format="dmol-arc")
        with open(allstrfile, "r") as f:
            all_es = list((round(float(line.strip().split()[-1]),6) for line in f.readlines() if "Energy" in line))
        with open(minstrfile, "r") as f:
            min_es = list((round(float(line.strip().split()[-2]),6) for line in f.readlines() if "Energy" in line))

        traj_strs = []
        add = traj_strs.append
        remove = min_es.pop

        for idx, e in enumerate(all_es):

            if min_es!=[] and e==min_es[0]:
                remove(0)

            elif e != -8888.88:
                all_strs[idx].info["ssw_energy"] = e
                add(all_strs[idx])

        assert min_es == []
        
        return traj_strs



class ModelDeviProcess():

    def __init__(
        self, 
        dirpath:str="./",
        bounds:list[float]|None=None,
        filename:str="model_devi.out",
        column:int=4
    ):

        self.column = column
        self.dirpath = dirpath
        self.devifile = path.join(dirpath, filename)
        self.bounds = self.gen_bounds() if bounds is None else bounds


    def gen_bounds(self, max_force_devi=None, hard_criterion:float=0.5):

        max_force_devi = self.get_max_force_devi() if max_force_devi is None else max_force_devi

        mean, sigma = get_mean_and_sigma(max_force_devi)
        bounds = [mean+2*sigma, mean+3*sigma]
        if mean > hard_criterion:
            bounds = [0.15, 0.3]

        elif bounds[0] > hard_criterion:
            bounds = [0.3, hard_criterion]

        elif bounds[1] > hard_criterion:
            bounds[1] = hard_criterion

        wflog.info("The bounds of this model deviation: [{0:0.2f}, {1:0.2f}]".format(bounds[0], bounds[1])) 
        return bounds


    @property
    def get_bounds(self):

        return self.bounds


    def get_devi_percent(self, max_force_devi=None, whether_to_write:bool=False):

        max_force_devi = self.get_max_force_devi() if max_force_devi is None else max_force_devi

        bounds = self.bounds
        accurate_mask = list(max_force_devi <= bounds[0])
        failed_mask = list(max_force_devi > bounds[1])
        candidate_mask = list((bounds[0] < max_force_devi) & ( max_force_devi <= bounds[1]))

        tot = len(max_force_devi)
        assert tot != 0

        percent = {
            "fail": round(failed_mask.count(True) / tot, 6),
            "candidate":  round(candidate_mask.count(True) / tot, 6),
            "accuracy": round(accurate_mask.count(True) / tot, 6),
            "max_force_devi": round(max_force_devi.max(), 6),
            "mean": round(np.mean(max_force_devi), 6)
        }

        if whether_to_write and not("*" in self.dirpath):
            with open(path.join(self.dirpath, "percent.json"), "w") as f:
                json.dump(percent, f, indent=4)
        
        else:
            return percent

    
    def get_max_force_devi(self):
        
        devifile = self.devifile
        column = self.column

        if "*" in devifile:
            max_force_devi = []
            for i in list((np.loadtxt(f)[:, column] for f in glob.glob(devifile))):
                max_force_devi.extend(i)

        else:
            assert path.isfile(devifile)
            max_force_devi = np.loadtxt(devifile)[:, column]
        
        return np.array(max_force_devi)


    def get_candidates(self, traj_strs:list[Atoms], max_force_devi=None, whether_to_write:bool=False):

        max_force_devi = self.get_max_force_devi() if max_force_devi is None else max_force_devi
        full_ids = np.arange(max_force_devi.size)

        bounds = self.bounds
        candidate_mask = (bounds[0] <= max_force_devi) & ( max_force_devi < bounds[1])

        assert len(traj_strs) == max_force_devi.size
        candidates = list(traj_strs[i] for i in full_ids[candidate_mask])

        if whether_to_write and not("*" in self.dirpath):
            write(path.join(self.dirpath, "candidates.xyz"), candidates, format="extxyz")
        else:
            return candidates



class ScreenProcess():

    def __init__(self, structures:list[Atoms]|None=None, filenames:list[str]|str|None=None, fmt:str|None=None, output_dir:str="./"):

        assert not(structures is None and filenames is None)

        if structures is None:
            filenames = filenames if isinstance(filenames, list) else [filenames]
            s = []
            for f in filenames:
                s.extend(read(f, ":", format=fmt))
            self.structures = s

        else:
            self.structures = structures

        self.output_dir=output_dir


    def run_filter_by_e(self, energies:list[float]|None=None, e_cutoff:int|float=0.1, whether_to_write:bool=True):

        structures = self.structures

        if energies:
            assert len(structures) == len(energies)
            energies = list((i[0]/i[1].get_global_number_of_atoms() for i in zip(energies, structures)))
            e_base = min(energies)
            screen_strs = list((structures[idx] for idx, e in enumerate(energies) if e-e_base < e_cutoff))

        else:
            e_base = min(list((i.get_total_energy()/i.get_global_number_of_atoms() for i in structures)))
            screen_strs = list((structure for structure in structures 
            if structure.get_total_energy()/structure.get_global_number_of_atoms()-e_base < e_cutoff))

        if whether_to_write:
            write(path.join(self.output_dir, "screen_structs_by_e.xyz"), screen_strs, format="extxyz")

        else:
            return  screen_strs


    @staticmethod
    def run_filter_by_cluster(
        fxyz,
        asap_setting:dict|None=None, 
        numb_structure_per_label:int=1, 
        noise_percent:float|int=100, 
        whether_to_write:bool=False, 
        whether_to_plot:bool=False,
        whether_to_remove:bool=False,
        **kwargs):
        
        assert isinstance(noise_percent, (float, int)) and noise_percent>=0

        if not(path.isfile(fxyz)) or len(read(fxyz, ":", format="extxyz")) == 0:
            return []
        
        else:
            output_dir = path.dirname(fxyz)
            callasap = CallASAP(
                fxyz=fxyz,
                user_setting=asap_setting, 
                output_dir=output_dir,
                **kwargs
            )
            
            structs = callasap.asapxyz.frames
            if len(structs) > 50:
                try:
                    label_ids = callasap.cluster(whether_to_plot=whether_to_plot)

                except ValueError as e:
                    wflog.info(f"{output_dir}: {e.args}")
                    label_ids = {1:list(range(len(structs)))}

                except Exception as e:
                    wflog.info(f"{output_dir}: {e.args}")
                    label_ids = {1:list(range(len(structs)))}

            else:
                label_ids = {-1:list(range(len(structs)))} 

            cluster_structures = []
            for i in label_ids:
                if i!=-1:
                    random.shuffle(label_ids[i])
                    if len(label_ids[i]) > numb_structure_per_label:
                        cluster_structures += list((structs[j] for j in label_ids[i][:numb_structure_per_label]))
                    else:
                        cluster_structures += list((structs[j] for j in label_ids[i]))

            noise = []
            if -1 in label_ids:
                random.shuffle(label_ids[-1])
                noise.extend([structs[i] for i in label_ids[-1][:int(len(label_ids[-1])*noise_percent/100)]])

            if whether_to_remove:
                f_remove(fxyz)
            
            if whether_to_write:
                write(path.join(output_dir, "cluster_structs.xyz"), cluster_structures+noise, format='extxyz')
            else:
                return cluster_structures+noise



class CandidateProcess():

    def __init__(
        self,
        dirpaths:str|list[str],
        output_file:str,
        accuracy:float|None=None,
        bounds:list[float]|None=None 
    ):

        self.bounds = self.__gen_bounds() if bounds is None else bounds
        self.accuracy = accuracy
        self.output_file = output_file
        self.dirpaths = dirpaths if isinstance(dirpaths, list) else glob.glob(dirpaths)


    def __gen_bounds(self, Devi=ModelDeviProcess):

        model_devis = []
        extend = model_devis.extend

        ret = list((extend(Devi(dirpath=dirpath).get_max_force_devi()) for dirpath in self.dirpaths))

        bounds = Devi.gen_bounds(max_force_devi=np.array(model_devis))

        return bounds


    def gen_candidates_ensemble(self, whether_to_return:bool=True):

        func = partial(self.candidates_indicator, whether_to_write=not(whether_to_return))

        if whether_to_return:
            results = parallel_by_threadpool(func, self.dirpaths, whether_to_return=whether_to_return)
            candidates_ensemble = list((result for result in results if result is not None))

            return candidates_ensemble
        
        else:
            parallel_by_threadpool(func, self.dirpaths)


    def candidates_indicator(self, dirpath:str, whether_to_write:bool=False, Traj=SSWTrajGeneration, Devi=ModelDeviProcess):

        accuracy = self.accuracy
        
        if accuracy is None:

            traj_strs = Traj(dirpath).gen_trajactory()
            candidates = Devi(dirpath, bounds=self.bounds).get_candidates(traj_strs, whether_to_write=whether_to_write)
            
            if not(whether_to_write):
                return candidates

        else:
            devi = Devi(dirpath, bounds=self.bounds)
            _accuracy = devi.get_devi_percent()["accuracy"]
            assert isinstance(accuracy, (float, int))

            if float(_accuracy) < accuracy:
                traj_strs = Traj(dirpath).gen_trajactory()
                candidates = devi.get_candidates(traj_strs, whether_to_write=whether_to_write)

                if not(whether_to_write):
                    return candidates
            
            else:
                wflog.info(f"The accuracy in the {path.basename(dirpath)} is larger than {accuracy:0.3f}, \
                so its candidate structures are not chosen for next process!")
    

    def filing_chosen_candidates(self, numb_candidates_per_traj:int=10, candidates:list[list]|None=None, whether_to_cluster:bool=True, Filter=ScreenProcess, **kwargs):

        if whether_to_cluster:
            self.gen_candidates_ensemble(whether_to_return=not(whether_to_cluster))
            func = partial(Filter.run_filter_by_cluster, **kwargs)
            files = list((path.join(dirpath, "candidates.xyz") for dirpath in self.dirpaths))
            results = list(parallel_by_pool(func, files, whether_to_return=True))

        else:
            candidates_ensemble = self.gen_candidates_ensemble() if candidates is None else candidates
            assert isinstance(candidates_ensemble[0], list)
            random.shuffle(candidates_ensemble)
            results = candidates_ensemble

        filing_chosen_candidates = []

        for i in results:
            if len(i) < numb_candidates_per_traj:
                filing_chosen_candidates.extend(i)
            
            else:
                filing_chosen_candidates.extend(i[:numb_candidates_per_traj])
        
        wflog.info(f"The total number of candidates is {len(filing_chosen_candidates)}.")
        write(self.output_file, filing_chosen_candidates, format="extxyz")



class DPProcess():

    @staticmethod
    def dptest_file_process(filename:str, dataset:str|None=None, peratom:bool=False, force_split:bool=False):
        
        data = np.loadtxt(filename)

        if peratom or force_split:
            assert path.isdir(dataset)

            with open(filename, "r") as f:
                system_dirs = list((path.join(dataset, path.basename(i.strip("# ").split(":")[0])) for i in f.readlines() if "#" in i))

            nframe = list((LabeledSystem(system_dir, fmt="deepmd/npy").get_nframes() for system_dir in system_dirs))
            natom = list((len(open(path.join(system_dir, "type.raw")).readlines()) for system_dir in system_dirs))

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
    

    def run_filter(self):

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
                system.to_deepmd_npy(f"{path.join(output_dir, file_name)}")
            
            else:
                system.to_deepmd_npy(f"{path.join(output_dir, system.formula)}")

