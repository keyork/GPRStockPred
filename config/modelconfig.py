
from models.rbfkernel import RBFKernelGPModel
from models.rqkernel import RQKernelGPModel
from models.maternkernel import MaternKernelGPModel
from models.spectraldeltakernel import SpectralDeltaKernelGPModel
from models.spectralmixturekernal import SpectralMixtureKernelGPModel

# DATA_PATH = './data/AMAZON COM INC (08-26-2021 _ 01-21-2022).csv'
# DATA_PATH = './data/APPLE INC (01-22-2021 _ 01-21-2022).csv'
DATA_PATH = './data/NVIDIA CORPORATION (01-22-2021 _ 01-21-2022).csv'


class RBFConfig:

    def __init__(self):

        self.data_path = DATA_PATH
        self.scale_train_test = 0.4
        self.train_step = [2000, 3000, 1000]
        self.learning_rate = 1
        
    def get_model(self, train_x, train_y, likelihood):
        model = RBFKernelGPModel(train_x, train_y, likelihood)
        return model


class RQConfig:

    def __init__(self):
    
        self.data_path = DATA_PATH
        self.scale_train_test = 0.4
        self.train_step = [5000, 3000, 1000]
        self.learning_rate = 1
        
    def get_model(self, train_x, train_y, likelihood):
        model = RQKernelGPModel(train_x, train_y, likelihood)
        return model


class MaternConfig:

    def __init__(self):
    
        self.data_path = DATA_PATH
        self.scale_train_test = 0.4
        self.train_step = [5000, 3000, 1000]
        self.learning_rate = 1
        
    def get_model(self, train_x, train_y, likelihood):
        model = MaternKernelGPModel(train_x, train_y, likelihood)
        return model


class SpectralDeltaConfig:

    def __init__(self):
        
        self.data_path = DATA_PATH
        self.scale_train_test = 0.4
        self.train_step = [5000, 4000, 1000]
        self.learning_rate = 0.1
        
    def get_model(self, train_x, train_y, likelihood, num_dims):
        model = SpectralDeltaKernelGPModel(train_x, train_y, likelihood, num_dims)
        return model


class SpectralMixtureConfig:

    def __init__(self):
        
        self.data_path = DATA_PATH
        self.scale_train_test = 0.4
        self.train_step = [2000, 3000, 3000]
        self.learning_rate = 0.1
        
    def get_model(self, train_x, train_y, likelihood, num_mixtures):
        model = SpectralMixtureKernelGPModel(train_x, train_y, likelihood, num_mixtures)
        return model