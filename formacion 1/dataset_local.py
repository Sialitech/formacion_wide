from clearml import Dataset

dataset = Dataset.create(
    dataset_name="boro_pilar",
    dataset_project="formacion2"
)
dataset.add_files(
    path="boro.csv",
)
dataset.upload()
dataset.finalize()
