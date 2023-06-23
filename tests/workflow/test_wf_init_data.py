import pytest
from shutil import rmtree
from checmate.workflow.wf_init_data import init_data_flow

config= {
    "general":{
        "dataset": "dataset",
        "type_map": ["Ga", "O"],
        "bad_data": "bad_data",
        "init": {
            "structure_paths": "../beta-Ga2O3.cif",
            "structure_format": "cif"
        }
    },
    "exploration":{
        "lasp":{
            "params": {
                "potential": "vasp",
                "SSW.SSWsteps": 2
            },
            "dpdispatcher":{
                "resources_temp_key": "cpu-lasp",
                "resources": {
                    "number_node": 2,
                    "cpu_per_node": 28,
                    "queue_name": "52-medium",
                    "group_size": 1
                },
                "task":{
                    "command": "mpiexec.hydra -genvall lasp"
                }
            }
        },
        "vasp":{
            "params":{
                "incar":{
                    "ENCUT":200
                }
            }
        }
    },
    "labeling":{
        "f_cutoff":10,
        "numb_structs":3,
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
                    "group_size": 1
                }
            }
        }
    }
}


#@pytest.mark.skip()
def test_init_data_flow1():

    init_data_flow(
        user_config=config
    )
#    rmtree("init.ssw_fp")


@pytest.mark.skip()
def test_init_data_flow2():

    init_data_flow(
        user_config=config,
        whether_to_ssw=False
    )
    rmtree("init.fp")
