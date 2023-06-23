import pytest
from ase.io import read
from shutil import rmtree

from checmate.unittask.fp import vasp_flow, cp2k_flow, vasp_nscf_flow, vasp_test, cp2k_test

structures = read("../beta-Ga2O3.cif",":","cif")
setting1 = {
    "params":{
        "incar":{
            "ENCUT":450,
            "NELM":10,
            "NPAR":4
        }
    },
    "dpdispatcher":{
        "resources":{
            "number_node": 1,
            "cpu_per_node": 24,
            "queue_name": "51-small",
            "group_size": 1

        },
        "resources_temp_key":"cpu-vasp"
    }
}
setting2 = {
    "params":{
        "FORCE_EVAL": {
            "DFT": {
                "MGRID": {"CUTOFF": 200},
                "SCF": {"MAX_SCF": 10}
            }
        } 
    },
    "dpdispatcher":{
        "resources":{
            "number_node": 1,
            "cpu_per_node": 24,
            "queue_name": "51-small",
            "group_size": 1

        },
        "resources_temp_key":"cpu-cp2k"
    }
}


#@pytest.mark.skip()
def test_fp_vasp():

    vasp_flow(
        user_config=setting1,
        structures=structures*2, 
        output_dir="test_fp_vasp", 
        whether_to_check=True
    )
    rmtree("test_fp_vasp")


#@pytest.mark.skip()
def test_fp_cp2k():
    
    cp2k_flow(
        user_config=setting2,
        structures=structures*2,
        output_dir="test_fp_cp2k",
        whether_to_check=True
    )
    rmtree("test_fp_cp2k")


#@pytest.mark.skip()
def test_nscf_vasp():

    vasp_nscf_flow(
        user_config=setting1,
        fpaths=["pre_calc"],
        output_dir="test_nscf_vasp" 
    )
    rmtree("test_nscf_vasp")


#@pytest.mark.skip()
def test_vasp_test():

    vasp_test(
        user_config={"vasp_config":setting1, "encuts":[450, 500], "k_r_densities":[80, 100]},
        structure=structures[0], 
        output_dir="vasp_test"
    )
    rmtree("vasp_test")


#@pytest.mark.skip()
def test_cp2k_test():

    cp2k_test(
        user_config={"cp2k_config":setting2, "cutoffs":[200, 250]},
        structure=structures[0], 
        output_dir="cp2k_test"
    )
    rmtree("cp2k_test")