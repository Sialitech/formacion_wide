# Descargue el conjunto de datos CIFAR y cree un conjunto de datos con la clase Dataset de wide
from clearml import StorageManager, Dataset
manager = StorageManager()

dataset_path = manager.get_local_copy(
    remote_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
)

dataset = Dataset.create(
    dataset_name="datasetremoto_pilar", dataset_project="formacion2"
)

# Prepare y limpie los datos aquí antes de agregarlos al conjunto de datos

dataset.add_files(path=dataset_path)

# El conjunto de datos se carga en el servidor WIDE de forma predeterminada
dataset.upload()

dataset.finalize()
