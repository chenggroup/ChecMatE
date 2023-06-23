import pytest
from shutil import rmtree
from checmate.workflow.wf_explore import explore_flow

config = {
    "general":{
        "init":{
            "structure_paths": "../beta-Ga2O3.cif",
            "structure_format": "cif"
        },
        "type_map": ["In", "Ga", "O"]
    },
    "exploration":{
        "lasp":{ 
            "params": {
                "potential": "lammps",
                "SSW.SSWsteps": 5
            },
            "dpdispatcher":{
                "resources_temp_key": "gpu3-lasp",
                "machine":{
                    "batch_type": "slurm"
                }
            }
        }, 
        "lammps":{
            "params":{
                "models": "../models/frozen_model*.pb"
            }
        }
    },
    "screening":{
        "bounds": [0.2, 0.4],
        "e_cutoff": 0.1,
        "numb_structure_per_label": 1
    },
    "labeling":{
        "vasp":{
            "params":{
                "incar":{
                    "ENCUT":450
                }
            },
            "dpdispatcher":{
                "resources_temp_key": "cpu-vasp",
                "resources":{
                    "number_node": 1,
                    "cpu_per_node": 28,
                    "queue_name": "52-medium",
                    "group_size": 10
                }
            }
        },
        "dp":{
            "dpdispatcher":{
                "resources_temp_key": "gpu3",
                "machine":{
                    "batch_type": "slurm"
                }
            }
        }
    }
}


@pytest.mark.skip()
def test_explore_flow1():

    explore_flow(
        user_config=config,
        whether_to_label=False
    )
    rmtree("explore.ssw")

#@pytest.mark.skip()
def test_explore_flow2():

#    config["exploration"]["lasp"]["params"]["models"] = ["../models/frozen_model1.pb", "../models/frozen_model2.pb"]

    explore_flow(
        user_config=config
    )
#    rmtree("explore.ssw_fp")
