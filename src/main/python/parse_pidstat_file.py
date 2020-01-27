import logging
import numpy as np
import sys

import pandas as pd


pidstat_file = sys.argv[1]

logging.basicConfig(level=logging.DEBUG
    , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

columns = ['time', 'UID', 'PID', 'minflt/s', 'majflt/s', 'VSZ', 'RSS', '%MEM'
    , 'Command']

dtypes = {'VSZ':np.float64, 'RSS':np.float64}

df = pd.read_csv(pidstat_file, sep='\s+', header=1, names=columns, dtype=dtypes)

logging.info("pidstat RSS, non-swapped physical memory peak usage: %0.2f GB"
    , df['RSS'].max()/(1024**2))

logging.info("pidstat VSZ, virtual memory peak usage: %0.2f GB"
    , df['VSZ'].max()/(1024**2))
