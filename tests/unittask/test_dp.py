import pytest
from shutil import rmtree
from checmate.unittask.dp import dptrain_flow, dptest_flow



#@pytest.mark.skip()
def test_dptrain():

    setting = {
        "numb_train": 1,
        "dataset": "../dataset",
        "type_map": ["Ga", "O", "In"],
        "dp_config": {
            "params":{
                "training": {
                    "numb_steps": 1000,
                }
            },
            "dpdispatcher":{
                "machine":{
                    "batch_type":"slurm",
                    "context_type": "LazyLocalContext"
                },
                "resources_temp_key":"gpu3"
            }
        }
    }

    dptrain_flow( 
        user_config=setting,
        output_dir="test_dptrain"
    )
    rmtree("test_dptrain")


#@pytest.mark.skip()
def test_dptest():

    setting = {
        "dataset": "../dataset",
        "models": ["../models/frozen_model1.pb"],
        "dp_config":{
            "dpdispatcher":{
                "machine":{
                    "batch_type":"slurm",
                    "context_type": "LazyLocalContext"
                },
                "resources_temp_key":"gpu3"
            }
        }
    }

    dptest_flow(
        user_config=setting,
        output_dir="test_dptest"
    )
    rmtree("test_dptest")
