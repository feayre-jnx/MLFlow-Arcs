#%%----------------------------

import subprocess
import os

## init options
gpu_no = "0"

## create the mlflow db folder if it does not exist
mlflow_db_path = './model_runs_db/'
if not os.path.exists(mlflow_db_path): os.makedirs(mlflow_db_path)

## set the environment to MLFLOW Track URI
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///./model_runs_db/model_runs.db"

experiments = [
        ## RBT TRAIN
        ("model_A.py", gpu_no, "1"),
        ("model_A.py", gpu_no, "2"),
        ("model_A.py", gpu_no, "3"),
]

for i in experiments:
    ## This catches all the printed text.
    #subprocess.run(["python", experiments[0], 12], capture_output=True, text=True)
    try:
        subprocess.run(["python", i[0], i[1], i[2]])
    except:
        print('ERROR: ', i[0])