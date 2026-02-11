from . import FairMethod
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo arbitrario
# TODO: Añadir que en el constructor se pueda personalizar este modelo (o pasar uno creado por el usuario)
class ModeloEnBruto(nn.Module):
    def __init__(self, tam_entrada):
        super().__init__()
        self.modelo = nn.Sequential(
                nn.Linear(tam_entrada, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64, 1))
    def forward(self, x):
        return self.modelo(x).squeeze(-1)

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
    
