import os
import ctypes
import numpy as np
from ctypes import pointer
from ase import Atom, Atoms
from ase.spacegroup import get_spacegroup
from functools import reduce
from dpdata import MultiSystems
from multiprocessing import Pool
from sys import getsizeof

class SingleAtom(object):

    def __init__(self, atom:Atom):
        self.xyz = atom.position
        self.ele = int(atom.number)
        self.elesymbol = atom.symbol
        self.force = []
        self.symf = []
        self.species = 0 
        self.bondlist = []
        self.bondtype = []
        self.Ctye = 'empty'
        self.charge = 0
        

class SingleAtoms(object):

    def __init__(self, struct:Atoms, if_force=False):
        self.struct = struct
        self.Lfor = if_force
        #self.Q = 0
        self.build_coord_set()
        #try:
        #    self.TransferToKplstr()
        #except:
        #    print(self.Energy)
        if if_force:
            self.build_force_set()


    def build_coord_set(self):

        self.Energy = float(self.struct.get_total_energy())
        self.Cell = self.struct.get_cell().array.tolist()
        self.Latt = self.struct.cell.cellpar().tolist()
        self.Coord = self.struct.get_positions().tolist()
        self.Nat = self.struct.get_global_number_of_atoms()
        self.EleNam = self.struct.get_chemical_symbols()
        self.iza = self.struct.get_atomic_numbers().tolist()
        self.sp = {}
        self.sporder = {}
        for idx, elenam in enumerate(sorted(set(self.EleNam), key=self.EleNam.index), start=1):
            self.sp[elenam] = self.EleNam.count(elenam)
            self.sporder[elenam] = idx
    

    def build_force_set(self):

        self.stress = self.struct.get_stress().tolist()
        self.For = self.struct.get_forces().tolist()
        self.maxF = max(abs(self.struct.get_forces().ravel()))
        self.maxFF = round(self.maxF, 4)
    

    def TransferToKplstr(self, sa=SingleAtom):

        self.atom = []
        for i in self.struct:
            self.atom.append(sa(i))
            if self.Lfor:
                self.atom[i].force= self.For[i]
        self.abc = self.Latt
        self.energy = self.Energy
        self.atom.sort(key =lambda X: X.ele)
        self.natom = self.Nat
        self.eleList, index, self.natompe = list(np.unique(
            [atom.ele for atom in self.atom], return_index=True, return_counts=True))
        self.nele = len(self.eleList)
        self.elenameList = [self.atom[i].elesymbol for i in index]
        self.lat = self.Cell


    def SteinhartQ_cal(self):

        q = ctypes.cdll.LoadLibrary(path.join(Path(__file__).resolve().parent, "calQ.so"))
        xa = reduce(lambda a,b:a+b, self.Coord)
        cell = reduce(lambda a,b:a+b, self.Cell)

        natom = pointer(ctypes.c_int(self.Nat))
        el    = pointer(ctypes.c_double(self.Energy))
        atom  = pointer(ctypes.c_int(0))
        sym   = pointer(ctypes.c_int(int(0)))
        za    = pointer((ctypes.c_int*len(self.iza))(*self.iza))
        coord = pointer((ctypes.c_double*len(xa))(*xa))
        rv    = pointer((ctypes.c_double*len(cell))(*cell))
        qglobal = pointer((ctypes.c_double*4)(*[0.0,0.0,0.0,0.0]))
        q.get_order_parameter_(natom, za, rv, coord, atom, qglobal, el, sym)
        qval = (qglobal.contents)[1:4]

        q.get_dis_weight_order_parameter_(natom, za, rv, coord, atom, qglobal, el, sym)
        qval2 = (qglobal.contents)[1:4]

        return (self.Energy, qval[0], qval[1], qval[2],qval2[0], qval2[1], qval2[2], \
            get_spacegroup(self.struct, 0.6).symbol, get_spacegroup(self.struct, 0.6),)


class MultiAtoms(object):

    all_atoms = []

    def __init__(self, ms:MultiSystems):

        self.structs = []
        for ls in ms:
            self.structs += ls.to_ase_structure()
        self.allatoms()
    

    def allatoms(self, sas=SingleAtoms):

        all_atoms = []
        for s in self.structs:
            all_atoms.append(sas(s))
        self.set_all_atoms(all_atoms=all_atoms)

    @classmethod
    def set_all_atoms(cls, all_atoms):

        cls.all_atoms = all_atoms


def pararun(func, iterarg, ncore=4):

    pool = Pool(processes=ncore)
    results = []
    results.append(pool.map_async(func, iterarg).get())
    pool.close()
    pool.join()
    return results


def para_SteinhartQ_cal(task:SingleAtoms):

    return task.SteinhartQ_cal()


def add(list1, list2):

    return list1 + list2

class Qfile():

    def __init__(self, ms:MultiSystems):
        self.ms = ms


    def get_Q(self, ncore=50, MA=MultiAtoms):

        all_EQ = []
        all_atoms = MA(self.ms).all_atoms[:30]
        if ncore == 1:
            ncore = 50
        size = ncore if len(all_atoms) >= ncore else len(all_atoms)
        part_atoms = [all_atoms[i:i+size] for i in range(0, len(all_atoms), size)]
        for part in part_atoms:
            if ncore > 1:
                results = pararun(func=para_SteinhartQ_cal, iterarg=part, ncore=ncore)
                part_EQ = reduce(add, results)
            else:
                part_EQ = [atoms.SteinhartQ_cal() for atoms in part]
            all_EQ += part_EQ
        return all_EQ


    def make_file(self, ncore=50, dirname='.', filename='QE.out'):
        
        all_EQ = self.get_Q(ncore=ncore)
        with open(os.path.join(dirname, filename), 'w') as fo:
            fo.write("# No.  Energy\tStein-Qvalue Minimum: Q2   Q4   Q6\tDistance-weighted Stein-Q value Minimum: Q2   Q4   Q6\t Spacegroup&id\n")
            for idx, EQs in enumerate(all_EQ, start=1):
                fo.write("%5d  %8.6f    %8.4f  %8.4f  %8.4f    %8.4f  %8.4f  %8.4f  %8s  %5d\n"%(idx, EQs[0],\
                    float(EQs[1]), float(EQs[2]), float(EQs[3]),\
                    float(EQs[4]), float(EQs[5]), float(EQs[6]),\
                    str(EQs[7]), int(EQs[8])))
        



    
    
