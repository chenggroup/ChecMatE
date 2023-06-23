import os
import logging


NAME="checmate"
SHORT_CMD="checmate"
wflog = logging.getLogger(__name__)
wflog.setLevel(logging.INFO)
wflogf = logging.FileHandler(os.getcwd()+os.sep+SHORT_CMD+".log", delay=True)
wflogf_formatter=logging.Formatter("CHECMATE %(levelname)s : %(message)s")
wflogf.setFormatter(wflogf_formatter)
wflog.addHandler(wflogf)