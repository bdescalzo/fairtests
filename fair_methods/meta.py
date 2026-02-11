from . import FairMethod
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import F
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo arbitrario
# TODO: Añadir que en el constructor se pueda personalizar este modelo (o pasar uno creado por el usuario)
class ModeloEnBruto(nn.Module):
    def __init__(self, tam_entrada):  # tam_entrada es 9 si no hay raza, 10 si la raza se añade
        super().__init__()
        self.fc1 = nn.Linear(tam_entrada, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
            x = F.relu(x)
            x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
            x = F.relu(x)
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x.squeeze(-1)

class MetaLearning(FairMethod):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.datos_cargados = False

	def load_data(self, dataloader_train, dataloader_test):
		self.dataloader_train = dataloader_train
		self.dataloader_test = dataloader_test
		self.datos_cargados = True
            
	def fit(self, sensitive_labels):
		if (not self.datos_cargados):
			print("No hay datos de entrenamiento cargados")
			return
		
    
