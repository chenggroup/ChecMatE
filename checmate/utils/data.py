import glob
import numpy as np
from os import path
from datetime import datetime

from ase.io import read
from dpdata import LabeledSystem, MultiSystems

from .. import wflog


class FPToDpdata():
    def __init__(self, dirname:str, filename:str, fmt="vasp", type_map=None):

        self.dirname = dirname
        self.filename = filename
        self.fmt = "auto" if fmt=="vasp" else "cp2k/output"
        self.type_map = type_map

    @property
    def get_dpdata_systems(self, LS=LabeledSystem, MS=MultiSystems):

        ms = MS()
        add = ms.append
        
        for i in glob.glob(path.join(self.dirname, "**", self.filename)):

            if self.filename == "OUTCAR":
                with open(i, "r") as f:
                    if "General timing and accounting informations" not in f.read()[-100:]:
                        wflog.info(f"WARNNING: The file {i} can't be added to dataset, because it doesn't finish successfully!")
                        continue

            elif self.filename == "cp2k.out":
                with open(i, "r") as f:
                    if "GPROGRAM ENDED AT" not in f.read()[-100:]:
                        wflog.info(f"WARNNING: The file {i} can't be added to dataset, because it doesn't finish successfully!")
                        continue
            try:
                ls = LS(i, type_map=self.type_map, fmt=self.fmt)

                if len(ls) > 0:
                    ms.append(ls)
                else:
                    wflog.info(f"WARNNING: The file {i} can't be added to dataset!")

            except:
                wflog.info(f"WARNNING: The file {i} can't be labelled by dpdata!")

        return ms



class LaspToDpdata():
    def __init__(
        self, 
        dirname: str,
        force: bool=True, 
        virial: bool=False
    ):
        self.dirname = dirname
        self.force = force
        self.virial = virial


    def gen_datas(self, strfile:str="allstr.arc", forfile:str="allfor.arc"):
        """
        processing structure and force files to obtain relevant data.

        Parameters:
            strfile: structure file (Lasp) of which the structure format is arc.
            forfile: force file (Lasp).
        """

        strs = read(strfile, index=":", format="dmol-arc")

        with open(strfile, "r") as f:
            energies = list((round(float(line.strip().split()[-1]),6) for line in f.readlines() if "Energy" in line))
        assert len(strs)==len(energies)

        if self.force:
            with open(forfile, "r") as f:
                fors = f.read().strip().split(' For')
            assert len(strs)==len(energies)

        datas = []
        for idx, struct in enumerate(strs):
    
            data = {}
            data["struct"] = struct
            data["energy"] = energies[idx]
            
            if self.force:
                data["force"] = []
                txt = fors[idx].strip("\n ").split("\n")
                assert np.isclose(data["energy"], eval(txt[0].strip().split()[-1]), atol=1e-6)
                for j in txt[2:]:
                    data["force"].append(np.array(j.strip().split(), dtype=float))

            if self.virial:
                data["virial"] = list(np.array(txt[1].strip().split(), dtype=float)*struct.get_volume())
    
            datas.append(data)
        return datas


    @staticmethod
    def system_info(data, type_idx_zero = True):
        """
        processing data to generate corresponding system information which is used to 
        generate LabeledSystem by dpdata.
        """

        symbols = data["struct"].get_chemical_symbols()
        atom_names = list((i for i in set(symbols)))
        atom_numbs = list((symbols.count(i) for i in set(symbols)))
        
        atom_types = []
        for idx, ii in enumerate(atom_numbs) :
            atom_types += [idx]*ii if type_idx_zero else [idx+1]*ii

        coord = data["struct"].get_positions()
        cell = np.array(data["struct"].get_cell())

        system = {
            'atom_numbs': atom_numbs,
            'atom_names': atom_names,
            'atom_types': np.array(atom_types),
            'orig': np.array([0, 0, 0]),
            'cells': np.array([cell]),
            'coords': np.array([coord]),
            'energies': np.array([data["energy"]])
        }

        if "force" in data:
            system["forces"] = np.array([data["force"]])

        if 'virial' in data:
            tmp_v = data['virial']
            virial = np.zeros([3,3])
            virial[0][0] = tmp_v[0]
            virial[1][1] = tmp_v[1]
            virial[2][2] = tmp_v[2]
            virial[0][1] = tmp_v[3]
            virial[1][0] = tmp_v[3]
            virial[1][2] = tmp_v[4]
            virial[2][1] = tmp_v[4]
            virial[0][2] = tmp_v[5]
            virial[2][0] = tmp_v[5]
            system['virials'] = np.array([virial])

        return system


    # we assume that the force is printed ...
    def get_dpdata_systems(self, strfile:str="allstr.arc", forfile:str="allfor.arc", LS=LabeledSystem, MS=MultiSystems, **kargs):
        """
        generating dataset by dpdata.

        Parameters:
            strfile: structure file (Lasp) of which the structure format is arc.
            forfile: force file (Lasp).
            virial: whether to include virial information.
            kargs: other parameters supported by LabeledSystem.
        """

        ms = MS()
        add = ms.append

        for fpath in glob.glob(self.dirname):

            sfile = path.join(fpath, strfile)

            if self.force:
                ffile = path.join(fpath, forfile)
                datas = self.gen_datas(sfile, ffile)
            else:
                datas = self.gen_datas(sfile)
            
            for data in datas:
                system_info = self.system_info(data)
                ls = LS(data=system_info, **kargs)
                add(ls)

        return ms



class DpdataToLasp():

    def __init__(self, dirname:str|None=None, filename:str|None=None, fmt:str='deepmd/npy', output_dir:str='./', ms:MultiSystems|None=None):

        self.dirname = dirname
        self.filename = filename
        self.fmt = fmt
        self.output_dir = output_dir
        self.ms = ms

        assert not(dirname is None and filename is None) or not(ms is None)


    def gen_structs(self):

        ms = MultiSystems.from_dir(self.dirname, self.filename, self.fmt) if self.ms is None else self.ms

        structs = []
        for ls in ms:
            structs += ls.to_ase_structure()

        return structs
    

    def gen_str_txt(self, idx, struct):

        strtxt = "\t\t\t\tEnergy\t%8d        0.0000 %17.6f\n"%(idx, struct.get_total_energy())
        strtxt += '!DATE     %s\n' % datetime.now().strftime('%b %d %H:%m:%S %Y')
        strtxt += 'PBC%15.9f%15.9f%15.9f%15.9f%15.9f%15.9f\n' % tuple(struct.cell.cellpar())

        for i, atom in enumerate(struct):
            strtxt += "%-2s%18.9f%15.9f%15.9f CORE %4d %-2s %-2s %8.4f %4d\n" % (
                        atom.symbol, atom.position[0], atom.position[1], atom.position[2],
                        i+1, atom.symbol, atom.symbol, 0.0, i+1)
        strtxt += "end\nend\n"

        return strtxt


    def gen_for_txt(self, struct):

        fortxt = " For   0  0  SS-fixlat   %12.6f\n" % (struct.get_total_energy())
        fortxt += "%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n" % tuple(struct.get_stress())

        for f in struct.get_forces():
            fortxt += "%15.8f %15.8f %15.8f\n" % (f[0], f[1], f[2])
        fortxt += "\n "

        return fortxt


    @property
    def make_arcfiles(self):

        structs = self.gen_structs()

        allstr_txt = '!BIOSYM archive 3\nPBC=ON\n'
        allfor_txt = ''
        for idx, s in enumerate(structs):
            allstr_txt += self.gen_str_txt(idx,s)
            allfor_txt += self.gen_for_txt(s)

        output_dir = self.output_dir
        with open(path.join(output_dir,'allstr.arc'),'w') as f_str:
            f_str.write(allstr_txt)

        with open(path.join(output_dir,'allfor.arc'),'w') as f_for:
            f_for.write(allfor_txt)



class TrajToDpdata():

    def __init__(self, dirname:str, step:int=1, fmt:str='lasp', strfile:str='traj.xyz', efile:str='energy.log'):

        self.step = step
        self.strfile = path.join(dirname, strfile)
        self.efile = path.join(dirname, efile)

        if fmt == 'lasp':
            datas = self.lasp2dpdata()

        elif fmt == 'lammps':
            datas = self.lammps2dpdata()

        else:
            raise TypeError("unknown format type: {}! Optional: lasp or lammps".format(fmt))

        self.datas = datas
        

    def lasp2dpdata(self):

        strs = read(self.strfile, index=':', format='extxyz')
        
        datas = []
        for i in range(0, len(strs), self.step):
            data = {}
            data['struct'] = strs[i]
            data['energy'] = strs[i].info["ssw_energy"]
            data['force'] = []
            datas.append(data)

        return datas


    def lammps2dpdata(self):

        
        strs = read(self.strfile, index=':', format='lammps-dump-text')
        energies = np.loadtxt(self.efile)[:, 3]

        datas = []
        for i in range(len(strs)):
            data = {}
            data['struct'] = strs[i]
            data['energy'] = float(energies[i*self.step])
            data['force'] = []
            datas.append(data)

        return datas
    
    @property
    def get_dpdata_systems(self, MS=MultiSystems, LD=LaspToDpdata):

        ms = MS()
        add = ms.append

        for data in self.datas:
            system_info = LD.system_info(data)
            ls = LabeledSystem(data=system_info)
            add(ls)

        return ms

        
