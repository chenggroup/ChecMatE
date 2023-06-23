import pytest
from ase.io import read
from shutil import rmtree

from checmate.unittask.ssw import fp_ssw_flow, lammps_ssw_flow

structures1 = read("../beta-Ga2O3.cif",":","cif")
structures2 = read("ssw_eam/lasp.str",":","dmol-arc")
setting1 = {
    "lasp_config":{
        "params": {
            "potential": "vasp",
            "SSW.SSWsteps": 0
        },
        "dpdispatcher":{
            "resources_temp_key": "cpu-lasp",
            "resources":{
                "number_node": 1,
                "cpu_per_node": 32,
                "queue_name": "53-medium"
            }
        }
    },
    "potential_config":{
        "params": {
            "incar":{
                "ENCUT": 200,
                "EDIFF": 1e-4
            }
        }
    }
}

setting2 = {
    "lasp_config":{
        "params": {
            "potential": "lammps",
            "SSW.SSWsteps": 2
        },
        "dpdispatcher":{
            "machine":{
                "batch_type":"slurm"
            },
            "resources_temp_key": "gpu3-lasp"
        }
    },
    "potential_config":{
        "params":{
            "type_map": ["Ga", "In", "O"],
            "models": ["../models/frozen_model1.pb"]
        }
    }
}

setting3 = {
    "lasp_config":{
        "params": {
            "potential": "lammps",
            "SSW.SSWsteps": 2
        },
        "dpdispatcher":{
            "resources_temp_key": "cpu-lasp",
            "resources":{
                "number_node": 1,
                "cpu_per_node": 32,
                "queue_name": "53-medium"
            },
            "forward_common_files":["ssw_eam/CuZr.eam"]
        }
    },
    "potential_config":{
        "template_path":"ssw_eam/in.simple",
        "params":{
            "type_map": ["Zr", "Cu"]
        }
    }
}


@pytest.mark.skip()
def test_vasp_ssw():

    fp_ssw_flow(
        structures=structures1,
        user_config=setting1,
        output_dir="test_vasp_ssw"
    )
    rmtree("test_vasp_ssw")

@pytest.mark.skip()
def test_dp_ssw():

    lammps_ssw_flow( 
        user_config=setting2,
        structures=structures1*2,
        output_dir="test_dp_ssw"
    )
    rmtree("test_dp_ssw")


@pytest.mark.skip()
def test_lammps_ssw():

    lammps_ssw_flow(
        structures=structures2,
        user_config=setting3,
        output_dir="test_lammps_ssw"
    )
    rmtree("test_lammps_ssw")
