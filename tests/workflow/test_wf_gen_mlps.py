import pytest
from checmate.pretask.sets import load_config
from checmate.workflow.wf_gen_mlps import gen_mlps_flow


#@pytest.mark.skip()
def test_gen_mlps_flow():
    gen_mlps_flow(
        user_config=load_config("gen_mlps_setting.json"),
        max_iter=3
    )

