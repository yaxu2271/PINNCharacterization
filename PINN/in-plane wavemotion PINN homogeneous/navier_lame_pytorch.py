import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm

np.random.seed(1234)

from pdb import set_trace as st
import pandas as pd

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
if not os.path.exists('./model/'):
    os.mkdir('./model/')

if not os.path.exists('./result/'):
    os.mkdir('./result/')

if not os.path.exists('./logs/'):
    os.mkdir('./logs/')



def calc_derivitives(f, v):
    result = torch.autograd.grad(
        f, v,
        grad_outputs=torch.ones_like(f),
        retain_graph=True,
        create_graph=True
    )[0]

    return result




class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

# the physics-informed neural network
class PhysicsInformedNN():
    def __init__(self, X_real, u_real, layers): # , lb, ub



        # data
        self.x = torch.tensor(X_real[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X_real[:, 1:2], requires_grad=True).float().to(device)
        self.u_real = torch.tensor(u_real).float().to(device)

        # settings
        self.mu_ = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_ = torch.tensor([0.0], requires_grad=True).to(device)


        self.mu_ = torch.nn.Parameter(self.mu_)
        self.lambda_ = torch.nn.Parameter(self.lambda_)

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('mu_', self.mu_)
        self.dnn.register_parameter('lambda_', self.lambda_)


        LEARNING_RATE = 0.0005

        self.optimizer = torch.optim.Adam(
            self.dnn.parameters(),
            lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-6)


        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x, y):
        u_pred = self.dnn(torch.cat([x, y], dim=1))
        # st()
        return u_pred

    def net_f(self, x, y):
        """ The pytorch autograd version of calculating residual """



        u_pred = self.net_u(x, y)

        ux = u_pred[:,0]
        uy = u_pred[:,1]
        # st()


        ux_x = calc_derivitives(ux, x)
        ux_y = calc_derivitives(ux, y)
        uy_x = calc_derivitives(uy, x)
        uy_y = calc_derivitives(uy, y)

        ux_xx = calc_derivitives(ux_x, x)
        ux_xy = calc_derivitives(ux_x, y)
        ux_yy = calc_derivitives(ux_y, y)
        uy_xx = calc_derivitives(uy_x, x)
        uy_yy = calc_derivitives(uy_y, y)
        uy_yx = calc_derivitives(uy_y, x)


        return ux, ux_xx, ux_xy, ux_yy, uy, uy_xx, uy_yx, uy_yy

    def loss_func(self):
        w = 3.91
        rho_parameter = 1
        mu_parameter = self.mu_
        lambda_parameter = self.lambda_

        gt_ux = self.u_real[:,0]
        gt_uy = self.u_real[:,1]


        output_ux, output_ux_xx, output_ux_xy, output_ux_yy, output_uy, output_uy_xx, output_uy_yx, output_uy_yy = self.net_f(self.x, self.y)


        c1 = ((w**2) * output_ux.flatten()) + (mu_parameter * (output_ux_xx.flatten() + output_ux_yy.flatten())) + ((lambda_parameter+mu_parameter)*(output_ux_xx.flatten() + output_uy_yx.flatten()))
        c2 = ((w**2) * output_uy.flatten()) + (mu_parameter * (output_uy_xx.flatten() + output_uy_yy.flatten())) + ((lambda_parameter+mu_parameter)*(output_ux_xy.flatten() + output_uy_yy.flatten()))




        c1_loss = torch.mean(torch.abs(c1)**2)
        c2_loss = torch.mean(torch.abs(c2)**2)

        loss_gt =  torch.mean((torch.abs(gt_ux-output_ux))**2) + torch.mean((torch.abs(gt_uy-output_uy))**2) # + torch.mean(c1 + c2) ** 2



        l1 = 0.065

        loss_Navier = l1*(c1_loss + c2_loss)

        loss = loss_gt + loss_Navier


        return loss,loss_gt,loss_Navier, output_ux, output_ux_xx, output_ux_xy, output_ux_yy, output_uy, output_uy_xx, output_uy_yx, output_uy_yy, mu_parameter, lambda_parameter

    def train(self, nIter, gt_ux, gt_ux_xx, gt_ux_xy, gt_ux_yy, gt_uy, gt_uy_xx, gt_uy_yx, gt_uy_yy):
        self.dnn.train()





        plt.figure()
        plt.ion()

        loss_list = []
        loss_gt_list = []
        loss_Navier_list = []

        lambda_list = []
        mu_list = []
        for epoch in tqdm(range(nIter)):

            loss,loss_gt,loss_Navier, output_ux, output_ux_xx, output_ux_xy, output_ux_yy, output_uy, output_uy_xx, output_uy_yx, output_uy_yy, mu_parameter, lambda_parameter = self.loss_func()

            loss_list.append(float(loss.detach().cpu().numpy()))
            loss_gt_list.append(float(loss_gt.detach().cpu().numpy()))
            loss_Navier_list.append(float(loss_Navier.detach().cpu().numpy()))


            lambda_list.append(float(lambda_parameter.detach().cpu().numpy()))
            mu_list.append(float(mu_parameter.detach().cpu().numpy()))



            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            self.mu_.data.clamp_(0)
            self.lambda_.data.clamp_(0)






            if epoch % 10000 == 0:
                
                torch.save(self.dnn.state_dict(),'result/model_{}.pth'.format(epoch))
                print('Save Done')
                


                plt.figure('loss_total')
                plt.title('loss_total')
                plt.cla()
                plt.plot(np.log10(loss_list))
                plt.savefig('logs/loss_total.png')


                plt.figure('loss_gt')
                plt.title('loss_gt')
                plt.cla()
                plt.plot(np.log10(loss_gt_list))
                plt.savefig('logs/loss_gt.png')


                plt.figure('loss_Navier')
                plt.title('loss_Navier')
                plt.cla()
                plt.plot(np.log10(loss_Navier_list))
                plt.savefig('logs/loss_Navier.png')


                plt.figure('loss_gt&Navier')
                plt.title('loss_gt&Navier')
                plt.plot(np.log10(loss_gt_list),'r-.', label='loss_gt')
                plt.plot(np.log10(loss_Navier_list),'y--',label='loss_Navier')
                plt.legend()
                plt.savefig('logs/loss_gt&Navier.png')
                plt.show()

                plt.figure('loss_total&gt&Navier')
                plt.title('loss_total&gt&Navier')
                plt.plot(np.log10(loss_gt_list),'r-.', label='loss_gt')
                plt.plot(np.log10(loss_Navier_list),'y--',label='loss_Navier')
                plt.plot(np.log10(loss_list),'-',label='loss_total')
                plt.legend()
                plt.savefig('logs/loss_total&gt&Navier.png')
                plt.show()


                print('loss MSE')
                print(loss)



                pred_ux = output_ux.detach().cpu().numpy().reshape(40,40)
                pred_ux_xx = output_ux_xx.detach().cpu().numpy().reshape(40,40)
                pred_ux_xy = output_ux_xy.detach().cpu().numpy().reshape(40,40)
                pred_ux_yy = output_ux_yy.detach().cpu().numpy().reshape(40,40)

                pred_uy = output_uy.detach().cpu().numpy().reshape(40,40)
                pred_uy_xx = output_uy_xx.detach().cpu().numpy().reshape(40,40)
                pred_uy_yx = output_uy_yx.detach().cpu().numpy().reshape(40,40)
                pred_uy_yy = output_uy_yy.detach().cpu().numpy().reshape(40,40)


                ux_diff = np.subtract(pred_ux,gt_ux)
                mae_ux_diff = np.abs(ux_diff).mean()
                mse_ux_diff = np.square(ux_diff).mean()
                print('mse_ux_diff')
                print(mse_ux_diff)
                print('mae_ux_diff')
                print(mae_ux_diff)

                ux_xx_diff = np.subtract(pred_ux_xx,gt_ux_xx)
                mse_ux_xx_diff = np.square(ux_xx_diff).mean()
                mae_ux_xx_diff = np.abs(ux_xx_diff).mean()
                print('mse_ux_xx_diff')
                print(mse_ux_xx_diff)
                print('mae_ux_xx_diff')
                print(mae_ux_xx_diff)

                ux_xy_diff = np.subtract(pred_ux_xy,gt_ux_xy)
                mse_ux_xy_diff = np.square(ux_xy_diff).mean()
                mae_ux_xy_diff = np.abs(ux_xy_diff).mean()
                print('mse_ux_xy_diff')
                print(mse_ux_xy_diff)
                print('mae_ux_xy_diff')
                print(mae_ux_xy_diff)

                ux_yy_diff = np.subtract(pred_ux_yy,gt_ux_yy)
                mse_ux_yy_diff = np.square(ux_yy_diff).mean()
                mae_ux_yy_diff = np.abs(ux_yy_diff).mean()
                print('mse_ux_yy_diff')
                print(mse_ux_yy_diff)
                print('mae_ux_yy_diff')
                print(mae_ux_yy_diff)


                uy_diff = np.subtract(pred_uy,gt_uy)
                mse_uy_diff = np.square(uy_diff).mean()
                mae_uy_diff = np.abs(uy_diff).mean()
                print('mse_uy_diff')
                print(mse_uy_diff)
                print('mae_uy_diff')
                print(mae_uy_diff)

                uy_xx_diff = np.subtract(pred_uy_xx,gt_uy_xx)
                mse_uy_xx_diff = np.square(uy_xx_diff).mean()
                mae_uy_xx_diff = np.abs(uy_xx_diff).mean()
                print('mse_uy_xx_diff')
                print(mse_uy_xx_diff)
                print('mae_uy_xx_diff')
                print(mae_uy_xx_diff)

                uy_yx_diff = np.subtract(pred_uy_yx,gt_uy_yx)
                mse_uy_yx_diff = np.square(uy_yx_diff).mean()
                mae_uy_yx_diff = np.abs(uy_yx_diff).mean()
                print('mse_uy_yx_diff')
                print(mse_uy_yx_diff)
                print('mae_uy_yx_diff')
                print(mae_uy_yx_diff)

                uy_yy_diff = np.subtract(pred_uy_yy,gt_uy_yy)
                mse_uy_yy_diff = np.square(uy_yy_diff).mean()
                mae_uy_yy_diff = np.abs(uy_yy_diff).mean()
                print('mse_uy_yy_diff')
                print(mse_uy_yy_diff)
                print('mae_uy_yy_diff')
                print(mae_uy_yy_diff)






                pred_ux_data = pd.DataFrame(pred_ux)
                pred_ux_xx_data = pd.DataFrame(pred_ux_xx)
                pred_ux_xy_data = pd.DataFrame(pred_ux_xy)
                pred_ux_yy_data = pd.DataFrame(pred_ux_yy)
                pred_uy_data = pd.DataFrame(pred_uy)
                pred_uy_xx_data = pd.DataFrame(pred_uy_xx)
                pred_uy_yx_data = pd.DataFrame(pred_uy_yx)
                pred_uy_yy_data = pd.DataFrame(pred_uy_yy)


                pred_ux_data.to_csv("pred_ux_data.csv")
                pred_ux_xx_data.to_csv("pred_ux_xx_data.csv")
                pred_ux_xy_data.to_csv("pred_ux_xy_data.csv")
                pred_ux_yy_data.to_csv("pred_ux_yy_data.csv")
                pred_uy_data.to_csv("pred_uy_data.csv")
                pred_uy_xx_data.to_csv("pred_uy_xx_data.csv")
                pred_uy_yx_data.to_csv("pred_uy_yx_data.csv")
                pred_uy_yy_data.to_csv("pred_uy_yy_data.csv")


                plt.figure('ux')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux')
                plt.imshow(pred_ux,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux')
                plt.imshow(gt_ux,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux - pred_ux)/max(gt_ux)')
                plt.imshow(np.abs(gt_ux - pred_ux)/np.max(np.abs(gt_ux)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux - pred_ux)/np.max(np.abs(gt_ux)),interpolation='spline16'))
                plt.savefig('logs/ux.png')


                plt.figure('ux_xx')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux_xx')
                plt.imshow(pred_ux_xx,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux_xx))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux_xx')
                plt.imshow(gt_ux_xx,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_xx))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux_xx - pred_ux_xx)/max(gt_ux_xx)')
                plt.imshow(np.abs(gt_ux_xx - pred_ux_xx)/np.max(np.abs(gt_ux_xx)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_xx - pred_ux_xx)/np.max(np.abs(gt_ux_xx)),interpolation='spline16'))
                plt.savefig('logs/ux_xx.png')


                plt.figure('ux_xy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux_xy')
                plt.imshow(pred_ux_xy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux_xy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux_xy')
                plt.imshow(gt_ux_xy,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_xy))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux_xy - pred_ux_xy)/max(gt_ux_xy)')
                plt.imshow(np.abs(gt_ux_xy - pred_ux_xy)/np.max(np.abs(gt_ux_xy)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_xy - pred_ux_xy)/np.max(np.abs(gt_ux_xy)),interpolation='spline16'))
                plt.savefig('logs/ux_xy.png')



                plt.figure('ux_yy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux_yy')
                plt.imshow(pred_ux_yy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux_yy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux_yy')
                plt.imshow(gt_ux_yy,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_yy))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux_yy - pred_ux_yy)/max(gt_ux_yy)')
                plt.imshow(np.abs(gt_ux_yy - pred_ux_yy)/np.max(np.abs(gt_ux_yy)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_yy - pred_ux_yy)/np.max(np.abs(gt_ux_yy)),interpolation='spline16'))
                plt.savefig('logs/ux_yy.png')


                plt.figure('uy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy')
                plt.imshow(pred_uy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy')
                plt.imshow(gt_uy,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy - pred_uy)/max(gt_uy)')
                plt.imshow(np.abs(gt_uy - pred_uy)/np.max(np.abs(gt_uy)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy - pred_uy)/np.max(np.abs(gt_uy)),interpolation='spline16'))
                plt.savefig('logs/uy.png')


                plt.figure('uy_xx')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy_xx')
                plt.imshow(pred_uy_xx,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy_xx))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy_xx')
                plt.imshow(gt_uy_xx,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_xx))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy_xx - pred_uy_xx)/max(gt_uy_xx)')
                plt.imshow(np.abs(gt_uy_xx - pred_uy_xx)/np.max(np.abs(gt_uy_xx)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_xx - pred_uy_xx)/np.max(np.abs(gt_uy_xx)),interpolation='spline16'))
                plt.savefig('logs/uy_xx.png')


                plt.figure('uy_yx')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy_yx')
                plt.imshow(pred_uy_yx,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy_yx))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy_yx')
                plt.imshow(gt_uy_yx,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_yx))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy_yx - pred_uy_yx)/max(gt_uy_yx)')
                plt.imshow(np.abs(gt_uy_yx - pred_uy_yx)/np.max(np.abs(gt_uy_yx)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_yx - pred_uy_yx)/np.max(np.abs(gt_uy_yx)),interpolation='spline16'))
                plt.savefig('logs/uy_yx.png')


                plt.figure('uy_yy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy_yy')
                plt.imshow(pred_uy_yy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy_yy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy_yy')
                plt.imshow(gt_uy_yy,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_yy))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy_yy - pred_uy_yy)/max(gt_uy_yy)')
                plt.imshow(np.abs(gt_uy_yy - pred_uy_yy)/np.max(np.abs(gt_uy_yy)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_yy - pred_uy_yy)/np.max(np.abs(gt_uy_yy)),interpolation='spline16'))
                plt.savefig('logs/uy_yy.png')


                plt.figure('lambda')
                plt.title('lambda')
                plt.cla()
                plt.plot(lambda_list)
                plt.savefig('logs/lambda_list.png')


                plt.figure('mu')
                plt.title('mu')
                plt.cla()
                plt.plot(mu_list)
                plt.savefig('logs/mu_list.png')

                print('lambda_parameter')
                print(lambda_parameter)

                print('mu_parameter')
                print(mu_parameter)

                loss_list_prediction = pd.DataFrame(loss_list)
                loss_list_prediction.to_csv("logs/loss_list_prediction.csv")

                loss_gt_list_prediction = pd.DataFrame(loss_gt_list)
                loss_gt_list_prediction.to_csv("logs/loss_gt_list_prediction.csv")

                loss_Navier_list_prediction = pd.DataFrame(loss_Navier_list)
                loss_Navier_list_prediction.to_csv("logs/loss_Navier_list_prediction.csv")

                lambda_list_prediction = pd.DataFrame(lambda_list)
                lambda_list_prediction.to_csv("logs/lambda_list_prediction.csv")

                mu_list_prediction = pd.DataFrame(mu_list)
                mu_list_prediction.to_csv("logs/mu_list_prediction.csv")




                plt.pause(0.01)



        # Backward and optimize
        self.optimizer.step(self.loss_func)

        plt.figure('loss_total')
        plt.title('loss_total')
        plt.cla()
        plt.plot(np.log10(loss_list))
        plt.savefig('logs/loss_total.png')


        plt.figure('loss_gt')
        plt.title('loss_gt')
        plt.cla()
        plt.plot(np.log10(loss_gt_list))
        plt.savefig('logs/loss_gt.png')


        plt.figure('loss_Navier')
        plt.title('loss_Navier')
        plt.cla()
        plt.plot(np.log10(loss_Navier_list))
        plt.savefig('logs/loss_Navier.png')


        plt.figure('loss_gt&Navier')
        plt.title('loss_gt&Navier')
        plt.plot(np.log10(loss_gt_list),'r-.', label='loss_gt')
        plt.plot(np.log10(loss_Navier_list),'y--',label='loss_Navier')
        plt.legend()
        plt.savefig('logs/loss_gt&Navier.png')
        plt.show()

        plt.figure('loss_total&gt&Navier')
        plt.title('loss_total&gt&Navier')
        plt.plot(np.log10(loss_gt_list),'r-.', label='loss_gt')
        plt.plot(np.log10(loss_Navier_list),'y--',label='loss_Navier')
        plt.plot(np.log10(loss_list),'-',label='loss_total')
        plt.legend()
        plt.savefig('logs/loss_total&gt&Navier.png')
        plt.show()




        plt.figure('lambda')
        plt.title('lambda')
        plt.cla()
        plt.plot(lambda_list)
        plt.savefig('logs/lambda_list.png')


        plt.figure('mu')
        plt.title('mu')
        plt.cla()
        plt.plot(mu_list)
        plt.savefig('logs/mu_list.png')

        loss_list_prediction = pd.DataFrame(loss_list)
        loss_list_prediction.to_csv("logs/loss_list_prediction.csv")

        loss_gt_list_prediction = pd.DataFrame(loss_gt_list)
        loss_gt_list_prediction.to_csv("logs/loss_gt_list_prediction.csv")

        loss_Navier_list_prediction = pd.DataFrame(loss_Navier_list)
        loss_Navier_list_prediction.to_csv("logs/loss_Navier_list_prediction.csv")

        lambda_list_prediction = pd.DataFrame(lambda_list)
        lambda_list_prediction.to_csv("logs/lambda_list_prediction.csv")

        mu_list_prediction = pd.DataFrame(mu_list)
        mu_list_prediction.to_csv("logs/mu_list_prediction.csv")







def main():
    


    layers = [2, 20, 40, 20, 2]


    data = scipy.io.loadmat('homogeneous40by40source3.mat')

    list_of_keys = list(data.keys())
    list_of_keys.remove('__header__')
    list_of_keys.remove('__version__')
    list_of_keys.remove('__globals__')



    for each_key in list_of_keys:

        vars()[each_key] = np.array(data[each_key].flatten())


    X_nl = np.hstack((vars()['x'].flatten()[:,None], vars()['y'].flatten()[:,None]))

    u_xy_nl = np.hstack((vars()['ux'].flatten()[:,None], vars()['uy'].flatten()[:,None]))





    gt_ux = vars()['ux'].reshape(40,40)
    gt_ux_xx = vars()['ux_xx'].reshape(40,40)
    gt_ux_xy = vars()['ux_xy'].reshape(40,40)
    gt_ux_yy = vars()['ux_yy'].reshape(40,40)

    gt_uy = vars()['uy'].reshape(40,40)
    gt_uy_xx = vars()['uy_xx'].reshape(40,40)
    gt_uy_yx = vars()['uy_yx'].reshape(40,40)
    gt_uy_yy = vars()['uy_yy'].reshape(40,40)

    ####
    model = PhysicsInformedNN(X_nl, u_xy_nl, layers)

    ## define epoch to train

    Epoch = 200000
    model.train(nIter=Epoch,gt_ux=gt_ux, gt_ux_xx=gt_ux_xx, gt_ux_xy=gt_ux_xy, gt_ux_yy=gt_ux_yy, gt_uy=gt_uy, gt_uy_xx=gt_uy_xx, gt_uy_yx=gt_uy_yx, gt_uy_yy=gt_uy_yy)



    lambda_value = model.lambda_.detach().cpu().numpy()
    mu_value = model.mu_.detach().cpu().numpy()

    print('lambda_value:')
    print(lambda_value)

    print('mu_value:')
    print(mu_value)







if __name__ == '__main__':
    main()
