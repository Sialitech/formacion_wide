import os


import pandas as pd
import numpy as np
from PIL import Image
from clearml import Task



task = Task.init(project_name='Acciona-MADRID', task_name='Artifacts example')

df = pd.DataFrame(
    {
        'num_legs': [2, 4, 8, 0],
        'num_wings': [2, 0, 0, 0],
        'num_specimen_seen': [10, 2, 1, 8]
    },
    index=['falcon', 'dog', 'spider', 'fish']
)


task.register_artifact('train', df, metadata={'counting': 'legs', 'max legs': 69})
df.sample(frac=0.5, replace=True, random_state=1)
Task.current_task().get_registered_artifacts()['train'].sample(frac=0.5, replace=True, random_state=1)


task.upload_artifact('Pandas', artifact_object=df)

task.upload_artifact('local file', artifact_object=os.path.join('data_samples', 'dancing.jpg'))

task.upload_artifact('dictionary', df.to_dict())

task.upload_artifact('boro', artifact_object='boro.csv')



print(df)

# we are done
print('Done')
