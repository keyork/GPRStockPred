
from matplotlib.pyplot import get
from pip import main
import torch
import gpytorch

from config.modelconfig import *
from tools.processdata import read_data, get_data, res_data, dwt_data_ca, dwt_data_cd


def train(config, is_res):

    all_data = read_data(config.data_path)
    if is_res:
        train_x, train_y, test_x, test_y, draw_test_x, start_data = res_data(all_data, config.scale_train_test)
    else:
        train_x, train_y, test_x, test_y, draw_test_x = get_data(all_data, config.scale_train_test)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    try:
        model = config.get_model(train_x, train_y, likelihood)
    except:
        try:
            model = config.get_model(train_x, train_y, likelihood, num_dims=1)
        except:
            model = config.get_model(train_x, train_y, likelihood, num_mixtures=50)
    
    model.train()
    likelihood.train()
    
    for step in config.train_step:
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        config.learning_rate /= 10
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(step):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if (i+1) % 100 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, step, loss.item()))
            optimizer.step()
    
    if is_res:
        return model, likelihood, train_x, train_y, test_x, test_y, draw_test_x, start_data
    else:
        return model, likelihood, train_x, train_y, test_x, test_y, draw_test_x


def train_dwt_a(config):

    all_data = read_data(config.data_path)
    train_x, train_y, test_x, test_y, draw_test_x = dwt_data_ca(all_data, config.scale_train_test)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    try:
        model = config.get_model(train_x, train_y, likelihood)
    except:
        try:
            model = config.get_model(train_x, train_y, likelihood, num_dims=1)
        except:
            model = config.get_model(train_x, train_y, likelihood, num_mixtures=50)
    
    model.train()
    likelihood.train()
    
    for step in config.train_step:
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        config.learning_rate /= 10
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(step):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if (i+1) % 100 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, step, loss.item()))
            optimizer.step()

    return model, likelihood, train_x, train_y, test_x, test_y, draw_test_x

def train_dwt_d(config):
    
    all_data = read_data(config.data_path)
    train_x, train_y, test_x, test_y, draw_test_x = dwt_data_cd(all_data, config.scale_train_test)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    try:
        model = config.get_model(train_x, train_y, likelihood)
    except:
        try:
            model = config.get_model(train_x, train_y, likelihood, num_dims=1)
        except:
            model = config.get_model(train_x, train_y, likelihood, num_mixtures=50)
    
    model.train()
    likelihood.train()
    
    for step in config.train_step:
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        config.learning_rate /= 10
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(step):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if (i+1) % 100 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, step, loss.item()))
            optimizer.step()

    return model, likelihood, train_x, train_y, test_x, test_y, draw_test_x


# if __name__ == '__main__':
#     config = RBFConfig()
#     train(config)
    

