from clearml import Task



task = Task.init(project_name="formacion2", task_name="hyperparametros")

args = {
     'dataset_task_id': '',
     'dataset_url': '',
     'random_state': 42,
     'test_size': 0.2,
 }
 # almacenar argumentos, estos se cambiarán en la tarea clonada cuando se ejecute 
task.connect(args)
print('Arguments: {}'.format(args))
