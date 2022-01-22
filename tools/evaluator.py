
import torch

class Evaluator:

    def __init__(self, y_pred, y_gt, train_end):

        self.y_pred = y_pred
        self.y_gt = y_gt
        self.train_end = train_end
        self.length = y_gt.size()[0]

        self.y_pred_l = None
        self.y_gt_l = None

        self.an1_l = None
        self.an5_l = None
        self.an10_l = None
        self.nr5_l = None
        self.mft_l = 0

        self.an1_v = None
        self.an5_v = None
        self.an10_v = None
        self.nr5_v = None
        self.mft_v = 0

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def get_logic(self):

        roll_pred = torch.roll(self.y_pred, 1)
        roll_gt = torch.roll(self.y_gt, 1)

        self.y_pred_l = self.y_pred - roll_pred
        self.y_gt_l = self.y_gt - roll_gt

        self.y_pred_l[0] = self.y_pred[0] - self.train_end
        self.y_gt_l[0] = self.y_gt[0] - self.train_end

        self.y_pred_l = (self.y_pred_l > 0) * 1
        self.y_gt_l = (self.y_gt_l > 0) * 1


    def get_mse(self, pred_array, gt_array):

        pred = torch.autograd.Variable(pred_array)
        gt = torch.autograd.Variable(gt_array)

        mse_loss = self.loss_fn(pred.float(), gt.float())
        
        return mse_loss


    def get_anx(self):

        self.an1_l = (self.y_pred_l[0] == self.y_gt_l[0]) * 1.0
        self.an5_l = torch.sum((self.y_pred_l[0:5] == self.y_gt_l[0:5]) * 1) / 5.0
        self.an10_l = torch.sum((self.y_pred_l[0:10] == self.y_gt_l[0:10]) * 1) / 10.0
        
        self.an1_v = self.get_mse(self.y_pred[0:1], self.y_gt[0:1])
        self.an5_v = self.get_mse(self.y_pred[0:5], self.y_gt[0:5])
        self.an10_v = self.get_mse(self.y_pred[0:10], self.y_gt[0:10])

    def get_nr5(self, noise_result):

        noise_anm_l = torch.sum((noise_result == self.y_gt_l) * 1) / self.length
        clean_anm_l = torch.sum((self.y_pred_l == self.y_gt_l) * 1) / self.length
        self.nr5_l = noise_anm_l / clean_anm_l

        noise_anm_v = self.get_mse(noise_result, self.y_gt)
        clean_anm_v = self.get_mse(self.y_pred, self.y_gt)
        self.nr5_v = noise_anm_v / clean_anm_v

    def get_mft(self):
        
        wrong_num = 0
        sample_idx = torch.tensor(0)
        while sample_idx < self.length:
            if self.y_pred_l[sample_idx] == self.y_gt_l[sample_idx]:
                self.mft_l += 1
                sample_idx += 1
            else:
                if wrong_num <= 5:
                    wrong_num += 1
                else:
                    break

        sample_idx = torch.tensor(0)
        while sample_idx < self.length:
            if torch.abs(self.y_pred[sample_idx] - self.y_gt[sample_idx]) <= \
                                                    0.1 * self.y_gt[sample_idx]:
                self.mft_v += 1
                sample_idx += 1
            else:
                break

    def start_eval(self):

        self.get_logic()
        # ANx-X
        self.get_anx()
        # NR5-X
        # self.get_nr5(noise_result=noise_pred)
        # MFT-X
        self.get_mft()

    
    def eval_result(self):

        print('|AN1-L |' + str(self.an1_l.numpy()))
        print('|AN5-L |' + str(self.an5_l.numpy()))
        print('|AN10-L|' + str(self.an10_l.numpy()))
        # print('|NR5-L |' + str(self.nr5_l.numpy()))
        print('|MFT-L |' + str(self.mft_l))

        print('|AN1-V |' + str(self.an1_v.numpy()))
        print('|AN5-V |' + str(self.an5_v.numpy()))
        print('|AN10-V|' + str(self.an10_v.numpy()))
        # print('|NR5-V |' + str(self.nr5_v.numpy()))
        print('|MFT-V |' + str(self.mft_v))

        


# # debug
# if __name__ == '__main__':

#     train_end = torch.tensor(10)
#     pred = torch.tensor([1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2])
#     gt =   torch.tensor([1,2,3,4,5,4,3,2,1,2,3,4,5,4,3,2])
#     evaluator = Evaluator(pred, gt, train_end)
#     evaluator.get_logic()
#     evaluator.get_mft()
#     print(evaluator.mft_l)
#     print(evaluator.mft_v)
