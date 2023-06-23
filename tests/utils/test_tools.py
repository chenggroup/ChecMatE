import shutil, os
#from checmate.utils.tools.asap import CallASAP
from checmate.utils.tools.sumo import CallSumo
#from checmate.utils.tools.pmg import KpointsPath

def test_sumo(make_temp_dir):

    shutil.copy("./vasprun.xml", "temp_dir")
    shutil.copy("./KPOINTS", "temp_dir")
    CallSumo(fpath="temp_dir", user_setting=None).run_bandstats()

    assert os.path.isfile("temp_dir/sumo-bandstat.log")

def test_sumo2():
    CallSumo(fpath=".", user_setting=None).run_bandstats()