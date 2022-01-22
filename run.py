import torch
import gpytorch
from matplotlib import pyplot as plt
from config.modelconfig import *
from train import train, train_dwt_a, train_dwt_d
import copy
import pywt
from tools.evaluator import Evaluator
import time



def test(model, likelihood, test_x):

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        observed_pred = likelihood(model(test_x))
        # print(observed_pred.mean)

    return observed_pred

def draw(observed_pred, train_x, train_y, test_x, test_y, draw_test_x, is_res):
    # if is_res:
    #     return
    with torch.no_grad():

        f, ax = plt.subplots(1, 1, figsize=(14, 6))
        lower, upper = observed_pred.confidence_region()
        draw_max = torch.max(torch.max(train_y), torch.max(test_y)) + \
            0.3 * torch.abs(torch.max(torch.max(train_y), torch.max(test_y)))
        draw_min = torch.min(torch.min(train_y), torch.min(test_y)) - \
            0.3 * torch.abs(torch.min(torch.min(train_y), torch.min(test_y)))

        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        ax.plot(draw_test_x.numpy(), test_y.numpy(), 'r*')

        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')

        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim(draw_min, draw_max)
        # if is_dwt:
        #     ax.set_ylim(-10, 10)
        # elif is_res:
        #     ax.set_ylim(-20, 20)
        # else:
        #     ax.set_ylim(0, 200)
        ax.legend(['Train Data', 'Ground Truth', 'Test Result', 'Confidence'])
        plt.savefig('./img/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.png')
        plt.show()
        eval_gt = test_y
        eval_pred = observed_pred.mean[-test_y.size()[0]:]
        train_end = train_y[-1]
        eval_tools = Evaluator(eval_pred, eval_gt, train_end)
        eval_tools.start_eval()
        eval_tools.eval_result()

def i_res(observed_pred, train_x, train_y, test_x, test_y, draw_test_x, start_data):
    
    real_data = copy.deepcopy(start_data)
    real_train = copy.deepcopy(torch.tensor([real_data]))
    real_draw_gt = copy.deepcopy(torch.tensor([real_data]))
    real_gt = torch.tensor([])
    real_pred = copy.deepcopy(torch.tensor([real_data]))
    
    with torch.no_grad():
        
        pred_result = observed_pred.mean
        for res in train_y:
            real_data += res
            real_train = torch.cat((real_train, torch.tensor([real_data])))
            real_draw_gt = torch.cat((real_draw_gt, torch.tensor([real_data])))
        real_gt = torch.cat((real_gt, torch.tensor([real_data])))
        for res in test_y:
            real_data += res
            real_draw_gt = torch.cat((real_draw_gt, torch.tensor([real_data])))
            real_gt = torch.cat((real_gt, torch.tensor([real_data])))
        real_data = start_data
        for res in pred_result:
            real_data += res
            real_pred = torch.cat((real_pred, torch.tensor([real_data])))

        train_x = torch.cat((train_x, torch.tensor([train_x.size()[0]])))
        draw_test_x = torch.cat((draw_test_x, torch.tensor([test_x.size()[0]])))
        test_x = torch.cat((test_x, torch.tensor([test_x.size()[0]])))
        
        # print(draw_test_x)

        f, ax = plt.subplots(1, 1, figsize=(14, 6))
        lower, upper = observed_pred.confidence_region()

        ax.plot(train_x.numpy(), real_train.numpy(), 'k*')
        ax.plot(draw_test_x.numpy(), real_gt.numpy(), 'r*')

        ax.plot(test_x.numpy(), real_pred.numpy(), 'b')

        draw_max = torch.max(torch.max(real_train), torch.max(real_gt)) + \
            0.3 * torch.abs(torch.max(torch.max(real_train), torch.max(real_gt)))
        draw_min = torch.min(torch.min(real_train), torch.min(real_gt)) - \
            0.3 * torch.abs(torch.min(torch.min(real_train), torch.min(real_gt)))

        ax.set_ylim(draw_min, draw_max)
        # ax.set_ylim(0, 200)
        # ax.set_ylim(-20, 20)
        ax.legend(['Train Data', 'Ground Truth', 'Test Result'])
        plt.savefig('./img/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.png')
        plt.show()

        eval_gt = real_gt
        eval_pred = real_pred[-real_gt.size()[0]:]
        train_end = real_train[-1]
        eval_tools = Evaluator(eval_pred, eval_gt, train_end)
        eval_tools.start_eval()
        eval_tools.eval_result()

def i_dwt(pred_d, train_x_d, train_y_d, test_x_d, test_y_d, draw_test_x_d,\
            pred_a, train_x_a, train_y_a, test_x_a, test_y_a, draw_test_x_a):
    
    real_train = pywt.idwt(train_y_a.numpy(), train_y_d.numpy(), 'db1')
    real_gt = pywt.idwt(test_y_a.numpy(), test_y_d.numpy(), 'db1')

    with torch.no_grad():
        
        pred_result_d = pred_d.mean
        pred_result_a = pred_a.mean
        real_pred = pywt.idwt(pred_result_a.numpy(), pred_result_d.numpy(), 'db1')

        train_len = real_train.shape[0]
        train_x = torch.linspace(0, train_len-1, train_len)

        gt_len = real_gt.shape[0]
        gt_x = torch.linspace(train_len, train_len+gt_len-1, gt_len)

        pred_len = real_pred.shape[0]
        pred_x = torch.linspace(0, pred_len-1, pred_len)


        f, ax = plt.subplots(1, 1, figsize=(14, 6))

        ax.plot(train_x.numpy(), real_train, 'k*')
        ax.plot(gt_x.numpy(), real_gt, 'r*')

        ax.plot(pred_x.numpy(), real_pred, 'b')

        draw_max = torch.max(torch.max(torch.from_numpy(real_train)), torch.max(torch.from_numpy(real_gt))) + \
            0.3 * torch.abs(torch.max(torch.max(torch.from_numpy(real_train)), torch.max(torch.from_numpy(real_gt))))
        draw_min = torch.min(torch.min(torch.from_numpy(real_train)), torch.min(torch.from_numpy(real_gt))) - \
            0.3 * torch.abs(torch.min(torch.min(torch.from_numpy(real_train)), torch.min(torch.from_numpy(real_gt))))

        ax.set_ylim(draw_min, draw_max)
        # ax.set_ylim(0, 200)
        # ax.set_ylim(-20, 20)
        ax.legend(['Train Data', 'Ground Truth', 'Test Result'])
        plt.savefig('./img/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.png')
        plt.show()

        eval_gt = torch.from_numpy(real_gt)
        eval_pred = torch.from_numpy(real_pred[-torch.from_numpy(real_gt).size()[0]:])
        train_end = torch.from_numpy(real_train)[-1]
        eval_tools = Evaluator(eval_pred, eval_gt, train_end)
        eval_tools.start_eval()
        eval_tools.eval_result()


if __name__ == '__main__':

    is_res = 0
    is_dwt = 1
    '''
    RBF
    RQ
    Matern
    SpectralDelta
    SpectralMixture
    '''
    config = SpectralMixtureConfig()
    
    if is_dwt:
        config = SpectralMixtureConfig()
        config.train_step = [300, 2000, 1000]
        model_d, likelihood_d, train_x_d, train_y_d, test_x_d, test_y_d, draw_test_x_d = train_dwt_d(config)
        pred_d = test(model_d, likelihood_d, test_x_d)
        draw(pred_d, train_x_d, train_y_d, test_x_d, test_y_d, draw_test_x_d, is_res)

        config = SpectralMixtureConfig()
        config.learning_rate = 1
        model_a, likelihood_a, train_x_a, train_y_a, test_x_a, test_y_a, draw_test_x_a = train_dwt_a(config)
        pred_a = test(model_a, likelihood_a, test_x_a)
        draw(pred_a, train_x_a, train_y_a, test_x_a, test_y_a, draw_test_x_a, is_res)

        i_dwt(pred_d, train_x_d, train_y_d, test_x_d, test_y_d, draw_test_x_d,\
                pred_a, train_x_a, train_y_a, test_x_a, test_y_a, draw_test_x_a)
    else:
        if is_res:
            model, likelihood, train_x, train_y, test_x, test_y, draw_test_x, start_data = train(config, is_res)
        else:
            model, likelihood, train_x, train_y, test_x, test_y, draw_test_x = train(config, is_res)
        pred = test(model, likelihood, test_x)
        
        if is_res:
            draw(pred, train_x, train_y, test_x, test_y, draw_test_x, is_res)
            i_res(pred, train_x, train_y, test_x, test_y, draw_test_x, start_data)
        else:
            draw(pred, train_x, train_y, test_x, test_y, draw_test_x, is_res)
    
    torch.save(model_d.state_dict(), './trained/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'-d.pth')
    torch.save(model_a.state_dict(), './trained/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'-a.pth')