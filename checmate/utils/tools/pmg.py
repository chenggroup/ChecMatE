import random
from os import path

from ase.io import write

from pymatgen.io.cif import CifWriter
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core.periodic_table import Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation, RemoveSitesTransformation

class KpointsPath():

    def __init__(self, structure, fmt="line"):

        self.structure = AseAtomsAdaptor.get_structure(structure)
        self.fmt = fmt
        self.kpts = None
        self.weights = None
        self.labels = None

    
    def gen_kpoints(self, reciprocal_density=100, line_density=20):

        fmt = self.fmt

        if fmt == "line":
            self.__generate_k_line(line_density)

        elif fmt == "uniform":
            self.__generate_k_boltztrap(reciprocal_density)

        elif fmt == "both":
            self.__generate_k_boltztrap(reciprocal_density)
            self.__generate_k_line(line_density)

        else:
            raise TypeError(f"The unknown format {self.fmt} of generating the kpoints. Supported: line, uniform, and both.")
        
        return Kpoints(
            style=Kpoints.supported_modes.Reciprocal,
            num_kpts=len(self.kpts),
            kpts=self.kpts,
            kpts_weights=self.weights,
            labels=self.labels,
        )
    

    def __generate_k_boltztrap(self, reciprocal_density=100):

        structure = self.structure
        grid = Kpoints.automatic_density_by_vol(structure, reciprocal_density).kpts
        ir_kpts = SpacegroupAnalyzer(structure, symprec=0.1).get_ir_reciprocal_mesh(grid[0])

        kpts = []
        add1 = kpts.append
        weights = []
        add2 = weights.append

        for k in ir_kpts:
            add1(k[0])
            add2(int(k[1]))

        self.kpts = kpts if self.kpts is None else self.kpts.extend(kpts)
        self.weights = weights if self.weights is None else self.weights.extend(weights)

        if self.fmt == "both":
            labels = [None]*len(ir_kpts)
            self.labels = labels if self.labels is None else self.labels.extend(labels)

    
    def __generate_k_line(self, line_density=20):

        structure = self.structure
        spg_analy = SpacegroupAnalyzer(structure)
        primitive_standard_structure = spg_analy.get_primitive_standard_structure(international_monoclinic=False)
        kpath = HighSymmKpath(primitive_standard_structure)
        kpts, labels = kpath.get_kpoints(
            line_density=line_density, coords_are_cartesian=False
        )
        weights = [1] * len(kpts) if self.fmt=="line" else [0.0] * len(kpts)

        self.kpts = kpts if self.kpts is None else self.kpts.extend(kpts)
        self.weights = weights if self.weights is None else self.weights.extend(weights)
        self.labels = labels if self.labels is None else self.labels.extend(labels)



class MPData():

    def __init__(self, api_key):
        
        self.mp = MPRester(api_key)


    def get_data(self, target, prop=''):

        data = self.mp.get_data(target, data_type='vasp', prop=prop)

        return data


    def get_structs(self, target, whether_to_write=False, output_dir='./'):

        structs = self.mp.get_structures(target, final=True)

        if whether_to_write:
            for i, struct in enumerate(structs):
                CifWriter(struct).write_file(
                    path.join(output_dir,f'{i}-{struct.formula}.cif'))

        return structs


    def get_structs_from_mp_id(self, target, output_dir='./'):

        props = self.get_data(target, prop='material_id')

        for prop in props:
            mp_id = prop['material_id']
            struct = self.get_structs(mp_id)[0]

            CifWriter(struct).write_file(
                path.join(output_dir, f'{mp_id}-{struct.formula}.cif'))



class RandomSubGen():

    def __init__(self, structure, vac_symbol:str, sub_symbol:str):

        self.structure = AseAtomsAdaptor.get_structure(structure)
        self.vac_symbol = vac_symbol
        self.sub_symbol = sub_symbol


    def __random_substitute(self, numb_sub:int=1):

        assert isinstance(numb_sub, int)
        structure = self.structure
        vac_symbol = self.vac_symbol
        sub_symbol = self.sub_symbol

        vac_symbol_indexs = list(structure.indices_from_symbol(symbol=vac_symbol))
        assert len(vac_symbol_indexs) >= numb_sub
        random.shuffle(vac_symbol_indexs)

        index_species_map = {}
        for i in range(numb_sub):
            index_species_map[vac_symbol_indexs[i]] = sub_symbol
        
        sub_structure = ReplaceSiteSpeciesTransformation(
            indices_species_map=index_species_map).apply_transformation(structure)

        return AseAtomsAdaptor.get_atoms(sub_structure)


    def random_substitutes(self, numb_sub_list:list, output_file:str|None=None):

        sub_structures = []
        for i in numb_sub_list:
            if i==0:
                sub_structures.append(AseAtomsAdaptor.get_atoms(self.structure))
            else:
                sub_structures.append(self.__random_substitute(numb_sub=i))
        
        if output_file is not None:
            write(output_file, sub_structures, format="cif")
        
        else:
            return sub_structures
        
    
    def __random_substitute_by_os(self, common_oxidation_state:int, species_os:dict[str, float]):

        structure = self.structure
        vac_symbol = self.vac_symbol
        sub_symbol = self.sub_symbol
        #structure.add_oxidation_state_by_element(species_os)
        assert isinstance(common_oxidation_state, int)

        v_os, s_os = abs(species_os[vac_symbol]), abs(species_os[sub_symbol])
        v_n, s_n = common_oxidation_state//v_os, common_oxidation_state//s_os 
        assert not(common_oxidation_state%v_os) and not(common_oxidation_state%s_os)

        vac_symbol_indexs = list(structure.indices_from_symbol(symbol=vac_symbol))
        assert len(vac_symbol_indexs) >= v_n
        random.shuffle(vac_symbol_indexs)

        index_species_map = {}
        for i in range(s_n):
            index_species_map[vac_symbol_indexs[i]] = sub_symbol
        
        sub_structure = ReplaceSiteSpeciesTransformation(
            indices_species_map=index_species_map).apply_transformation(structure)

        try:
            sub_structure = RemoveSitesTransformation(
                indices_to_remove=vac_symbol_indexs[s_n:v_n]).apply_transformation(sub_structure)
        except:
            pass
        
        return AseAtomsAdaptor.get_atoms(sub_structure)


    def random_substitutes_by_os(self, common_oxidation_state_list:list, species_os:dict, output_file:str|None=None):

        sub_structures = []
        for i in common_oxidation_state_list:
            if i==0:
                sub_structures.append(AseAtomsAdaptor.get_atoms(self.structure))
            else:
                sub_structures.append(self.__random_substitute_by_os(i, species_os))
        
        if output_file is not None:
            write(output_file, sub_structures, format="cif")
        
        else:
            return sub_structures

        

        
    