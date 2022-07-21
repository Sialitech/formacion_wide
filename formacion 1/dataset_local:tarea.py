from clearml import Task, Dataset

task = Task.init(project_name="Acciona_MADRID", task_name="Tarea+Dataset")

dataset = Dataset.create(
    dataset_name="boro1_csv",
    dataset_project=task.get_project_name(),
    use_current_task=False,
)
dataset.add_files(
    path="boro.csv",
)
dataset.upload()
dataset.finalize()
