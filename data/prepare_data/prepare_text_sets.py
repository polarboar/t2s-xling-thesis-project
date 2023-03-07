import os

import jiant.utils.python.io as py_io
import jiant.proj.simple.runscript as simple_run
import jiant.scripts.download_data.runscript as downloader

TASK_NAME = "sst"

downloader.download_data([TASK_NAME], "/disk/scratch1/ramons/data/t2s-xling/sst/")



