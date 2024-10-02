import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# from plotting import newfig, savefig
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
    
    


def flatten_n_source_data(data, N_source):

    for i in range(N_source):
        if i == 0:
            re_ = data[i].flatten()[:,None]
        else:
            re_ = np.concatenate((re_,data[i].flatten()[:,None]),axis=0)

    return re_




def PrepareData(data_path, source_list):

    data = scipy.io.loadmat(data_path)
    N_source = len(source_list)

    ns, w, l = data['ux'].shape
    flatten_len = w*l

    ux = flatten_n_source_data(data['ux'], N_source)
    uy = flatten_n_source_data(data['uy'], N_source)
    
    ux_xx = flatten_n_source_data(data['ux_xx'], N_source)
    ux_xy = flatten_n_source_data(data['ux_xy'], N_source)
    ux_yy = flatten_n_source_data(data['ux_yy'], N_source)

    uy_xx = flatten_n_source_data(data['uy_xx'], N_source)
    uy_yx = flatten_n_source_data(data['uy_yx'], N_source)
    uy_yy = flatten_n_source_data(data['uy_yy'], N_source)

    x = flatten_n_source_data(data['x'], N_source)
    y = flatten_n_source_data(data['y'], N_source)


    s = np.repeat(source_list, flatten_len)[:,None]


    return (data, x, y, s, ux, uy, ux_xx, ux_xy, ux_yy, uy_xx, uy_yx, uy_yy)



def calc_derivitives(f, v):
    result = torch.autograd.grad(
        f, v, 
        grad_outputs=torch.ones_like(f),
        retain_graph=True,
        create_graph=True
    )[0]

    return result

def loop_auto_derivitives(nf, v):
    n = nf.shape[1]
    for i in range(n):
        if i == 0:
            re_ = calc_derivitives(nf[:,i], v)
        else:
            re_ = torch.cat((re_, calc_derivitives(nf[:,i], v)),dim=1)
    return re_


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
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
    def __init__(self, X_real, u_real, layers, source_num):
        

        
        # data
        self.source_num = source_num
        self.x = torch.tensor(X_real[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X_real[:, 1:2], requires_grad=True).float().to(device)
        self.s = torch.tensor(X_real[:, 2:3], requires_grad=True).float().to(device)
        self.u_real = torch.tensor(u_real).float().to(device)
        



        size_parameter = int(self.x.shape[0]/source_num)
  


        self.mu_ = torch.nn.Parameter(torch.zeros((size_parameter), requires_grad=True).to(device))
        self.lambda_ = torch.nn.Parameter(torch.zeros((size_parameter), requires_grad=True).to(device))



        self.mu_ = torch.nn.Parameter(self.mu_)
        self.lambda_ = torch.nn.Parameter(self.lambda_)

        
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('mu_', self.mu_)
        self.dnn.register_parameter('lambda_', self.lambda_)

  
        

        LEARNING_RATE = 0.0005


        
        self.optimizer = torch.optim.Adam(
            list(self.dnn.parameters()), 
            lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-6)

        self.iter = 0
        
    def net_u(self, x, y, s):  
        u_pred = self.dnn(torch.cat([x, y, s], dim=1))
        return u_pred
    
    def net_f(self, x, y, s):


        u_pred = self.net_u(x, y, s)
        ux = u_pred[:,0]
        uy = u_pred[:,1]


        
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


        mu_parameter = self.mu_.repeat(1,self.source_num)        
        lambda_parameter = self.lambda_.repeat(1,self.source_num)  



        gt_ux = self.u_real[:,0]
        gt_uy = self.u_real[:,1]

 



        output_ux, output_ux_xx, output_ux_xy, output_ux_yy, output_uy, output_uy_xx, output_uy_yx, output_uy_yy = self.net_f(self.x, self.y, self.s)


        c1 = ((w**2) * output_ux) + (mu_parameter * (output_ux_xx.squeeze() + output_ux_yy.squeeze())) + ((lambda_parameter+mu_parameter)*(output_ux_xx.squeeze()+ output_uy_yx.squeeze()))
        c2 = ((w**2) * output_uy) + (mu_parameter * (output_uy_xx.squeeze() + output_uy_yy.squeeze())) + ((lambda_parameter+mu_parameter)*(output_ux_xy.squeeze() + output_uy_yy.squeeze()))
        
        
        
        c1_loss = torch.mean(c1**2)
        c2_loss = torch.mean(c2**2)



        loss_gt =  torch.mean((gt_ux-output_ux)**2) + torch.mean((gt_uy-output_uy)**2) # + torch.mean(c1 + c2) ** 2
        
        l1 = 0.065
        
        loss_Navier = c1_loss + c2_loss

        loss = loss_gt + l1*loss_Navier
        

        
        return loss, loss_gt, loss_Navier, output_ux, output_ux_xx, output_ux_xy, output_ux_yy, output_uy, output_uy_xx, output_uy_yx, output_uy_yy




    def train(self, nIter, gt_ux, gt_ux_xx, gt_ux_xy, gt_ux_yy, gt_uy, gt_uy_xx, gt_uy_yx, gt_uy_yy, source_num):
        self.dnn.train()
        self.source_num = source_num



        

        plt.figure()
        plt.ion()

        loss_list = []
        loss_gt_list = []
        loss_Navier_list = []

        for epoch in tqdm(range(nIter)):
            
            loss, loss_gt, loss_Navier, output_ux, output_ux_xx, output_ux_xy, output_ux_yy, output_uy, output_uy_xx, output_uy_yx, output_uy_yy = self.loss_func()
            
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.mu_.data.clamp_(0)
            self.lambda_.data.clamp_(0)


            
            if epoch % 100000 == 0:
                torch.save(self.dnn.state_dict(),'result/model_{}.pth'.format(epoch))
                print('Save Done')
                
            if epoch % 20000 == 0:
        
                loss_list.append(float(loss.detach().cpu().numpy()))
                loss_gt_list.append(float(loss_gt.detach().cpu().numpy()))
                loss_Navier_list.append(float(loss_Navier.detach().cpu().numpy()))
                
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
                
                loss_list_prediction = pd.DataFrame(loss_list)
                loss_list_prediction.to_csv("logs/loss_list_prediction.csv")
                
                loss_gt_list_prediction = pd.DataFrame(loss_gt_list)
                loss_gt_list_prediction.to_csv("logs/loss_gt_list_prediction.csv")
                
                loss_Navier_list_prediction = pd.DataFrame(loss_Navier_list)
                loss_Navier_list_prediction.to_csv("logs/loss_Navier_list_prediction.csv")
                
                


            if epoch % 50000 == 0:
                # st()
                print('loss MSE')
                print(loss)
                
                
                mu_ = self.mu_.detach().cpu().numpy().reshape(40,40)
                lambda_ = self.lambda_.detach().cpu().numpy().reshape(40,40)
                
                print('-----------------------------------------------')
                print(f'Mu(1): {np.mean(mu_[:20,:20])}')
                print(f'Mu(2): {np.mean(mu_[:20,20:40])}')
                print(f'Mu(3): {np.mean(mu_[20:40,:20])}')
                print(f'Mu(4): {np.mean(mu_[20:40,20:40])}')
            
                print(f'Lam(1): {np.mean(lambda_[:20,:20])}')
                print(f'Lam(2): {np.mean(lambda_[:20,20:40])}')
                print(f'Lam(3): {np.mean(lambda_[20:40,:20])}')
                print(f'Lam(4): {np.mean(lambda_[20:40,20:40])}')
                
                print('-----------------------------------------------')
                print(f'Mu(1): {np.std(mu_[:20,:20])}')
                print(f'Mu(2): {np.std(mu_[:20,20:40])}')
                print(f'Mu(3): {np.std(mu_[20:40,:20])}')
                print(f'Mu(4): {np.std(mu_[20:40,20:40])}')
            
                print(f'Lam(1): {np.std(lambda_[:20,:20])}')
                print(f'Lam(2): {np.std(lambda_[:20,20:40])}')
                print(f'Lam(3): {np.std(lambda_[20:40,:20])}')
                print(f'Lam(4): {np.std(lambda_[20:40,20:40])}')






                pred_ux = output_ux.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)
                pred_ux_xx = output_ux_xx.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)
                pred_ux_xy = output_ux_xy.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)
                pred_ux_yy = output_ux_yy.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)

                pred_uy = output_uy.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)
                pred_uy_xx = output_uy_xx.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)
                pred_uy_yx = output_uy_yx.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)
                pred_uy_yy = output_uy_yy.detach().cpu().numpy()[:FLATTEN_LEN].reshape(40,40)




                ux_diff = np.subtract(pred_ux,gt_ux[:FLATTEN_LEN].reshape(40,40))
                mae_ux_diff = np.abs(ux_diff).mean()
                mse_ux_diff = np.square(ux_diff).mean()
                print('mse_ux_diff')
                print(mse_ux_diff)
                print('mae_ux_diff')
                print(mae_ux_diff)
                
                ux_xx_diff = np.subtract(pred_ux_xx,gt_ux_xx[:FLATTEN_LEN].reshape(40,40))
                mse_ux_xx_diff = np.square(ux_xx_diff).mean()
                mae_ux_xx_diff = np.abs(ux_xx_diff).mean()
                print('mse_ux_xx_diff')
                print(mse_ux_xx_diff)
                print('mae_ux_xx_diff')
                print(mae_ux_xx_diff)
                
                ux_xy_diff = np.subtract(pred_ux_xy,gt_ux_xy[:FLATTEN_LEN].reshape(40,40))
                mse_ux_xy_diff = np.square(ux_xy_diff).mean()
                mae_ux_xy_diff = np.abs(ux_xy_diff).mean()
                print('mse_ux_xy_diff')
                print(mse_ux_xy_diff)
                print('mae_ux_xy_diff')
                print(mae_ux_xy_diff)
                
                ux_yy_diff = np.subtract(pred_ux_yy,gt_ux_yy[:FLATTEN_LEN].reshape(40,40))
                mse_ux_yy_diff = np.square(ux_yy_diff).mean()
                mae_ux_yy_diff = np.abs(ux_yy_diff).mean()
                print('mse_ux_yy_diff')
                print(mse_ux_yy_diff)
                print('mae_ux_yy_diff')
                print(mae_ux_yy_diff)
                
                
                uy_diff = np.subtract(pred_uy,gt_uy[:FLATTEN_LEN].reshape(40,40))
                mse_uy_diff = np.square(uy_diff).mean()
                mae_uy_diff = np.abs(uy_diff).mean()
                print('mse_uy_diff')
                print(mse_uy_diff)
                print('mae_uy_diff')
                print(mae_uy_diff)
                
                uy_xx_diff = np.subtract(pred_uy_xx,gt_uy_xx[:FLATTEN_LEN].reshape(40,40))
                mse_uy_xx_diff = np.square(uy_xx_diff).mean()
                mae_uy_xx_diff = np.abs(uy_xx_diff).mean()
                print('mse_uy_xx_diff')
                print(mse_uy_xx_diff)
                print('mae_uy_xx_diff')
                print(mae_uy_xx_diff)
                
                uy_yx_diff = np.subtract(pred_uy_yx,gt_uy_yx[:FLATTEN_LEN].reshape(40,40))
                mse_uy_yx_diff = np.square(uy_yx_diff).mean()
                mae_uy_yx_diff = np.abs(uy_yx_diff).mean()
                print('mse_uy_yx_diff')
                print(mse_uy_yx_diff)
                print('mae_uy_yx_diff')
                print(mae_uy_yx_diff)
                
                uy_yy_diff = np.subtract(pred_uy_yy,gt_uy_yy[:FLATTEN_LEN].reshape(40,40))
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

                # st()

                pred_mu_parameter = pd.DataFrame(mu_)
                pred_lambda_parameter = pd.DataFrame(lambda_)
                

                
                pred_ux_data.to_csv("logs/pred_ux_data.csv")
                pred_ux_xx_data.to_csv("logs/pred_ux_xx_data.csv")
                pred_ux_xy_data.to_csv("logs/pred_ux_xy_data.csv")
                pred_ux_yy_data.to_csv("logs/pred_ux_yy_data.csv")
                pred_uy_data.to_csv("logs/pred_uy_data.csv")
                pred_uy_xx_data.to_csv("logs/pred_uy_xx_data.csv")
                pred_uy_yx_data.to_csv("logs/pred_uy_yx_data.csv")
                pred_uy_yy_data.to_csv("logs/pred_uy_yy_data.csv")

                pred_mu_parameter.to_csv("logs/pred_mu_parameter.csv")
                pred_lambda_parameter.to_csv("logs/pred_lambda_parameter.csv")
                


                plt.figure('parameters')
                plt.subplot(1,2,1)
                plt.cla()
                plt.title('mu')
                plt.imshow(mu_,interpolation='spline16', cmap='Blues')
                plt.colorbar(plt.imshow(mu_,interpolation='spline16', cmap='Blues'))
                plt.clim(0, 4) 
                plt.subplot(1,2,2)
                plt.cla()
                plt.title('lambda')
                plt.imshow(lambda_,interpolation='spline16', cmap='Blues')
                plt.colorbar(plt.imshow(lambda_,interpolation='spline16', cmap='Blues'))
                plt.clim(0, 3) 
                plt.savefig('logs/parameters.png')


                gt_ux_plot = gt_ux[:FLATTEN_LEN].reshape(40,40)
                
                plt.figure('ux')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux')
                plt.imshow(pred_ux,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux')
                plt.imshow(gt_ux_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux - pred_ux)/max(gt_ux)')
                plt.imshow(np.abs(gt_ux_plot - pred_ux)/np.max(np.abs(gt_ux_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_plot - pred_ux)/np.max(np.abs(gt_ux_plot)),interpolation='spline16'))
                plt.savefig('logs/ux.png')

                gt_ux_xx_plot = gt_ux_xx[:FLATTEN_LEN].reshape(40,40) 
                
                plt.figure('ux_xx')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux_xx')
                plt.imshow(pred_ux_xx,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux_xx))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux_xx')
                plt.imshow(gt_ux_xx_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_xx_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux_xx - pred_ux_xx)/max(gt_ux_xx)')
                plt.imshow(np.abs(gt_ux_xx_plot - pred_ux_xx)/np.max(np.abs(gt_ux_xx_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_xx_plot - pred_ux_xx)/np.max(np.abs(gt_ux_xx_plot)),interpolation='spline16'))
                plt.savefig('logs/ux_xx.png')

                gt_ux_xy_plot = gt_ux_xy[:FLATTEN_LEN].reshape(40,40) 

                plt.figure('ux_xy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux_xy')
                plt.imshow(pred_ux_xy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux_xy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux_xy')
                plt.imshow(gt_ux_xy_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_xy_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux_xy - pred_ux_xy)/max(gt_ux_xy)')
                plt.imshow(np.abs(gt_ux_xy_plot - pred_ux_xy)/np.max(np.abs(gt_ux_xy_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_xy_plot - pred_ux_xy)/np.max(np.abs(gt_ux_xy_plot)),interpolation='spline16'))
                plt.savefig('logs/ux_xy.png')


                gt_ux_yy_plot = gt_ux_yy[:FLATTEN_LEN].reshape(40,40) 

                plt.figure('ux_yy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_ux_yy')
                plt.imshow(pred_ux_yy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_ux_yy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_ux_yy')
                plt.imshow(gt_ux_yy_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_ux_yy_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_ux_yy - pred_ux_yy)/max(gt_ux_yy)')
                plt.imshow(np.abs(gt_ux_yy_plot - pred_ux_yy)/np.max(np.abs(gt_ux_yy_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_ux_yy_plot - pred_ux_yy)/np.max(np.abs(gt_ux_yy_plot)),interpolation='spline16'))
                plt.savefig('logs/ux_yy.png')


                gt_uy_plot = gt_uy[:FLATTEN_LEN].reshape(40,40) 
                
                plt.figure('uy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy')
                plt.imshow(pred_uy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy')
                plt.imshow(gt_uy_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy - pred_uy)/max(gt_uy)')
                plt.imshow(np.abs(gt_uy_plot - pred_uy)/np.max(np.abs(gt_uy_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_plot - pred_uy)/np.max(np.abs(gt_uy_plot)),interpolation='spline16'))
                plt.savefig('logs/uy.png')

                gt_uy_xx_plot = gt_uy_xx[:FLATTEN_LEN].reshape(40,40) 

                plt.figure('uy_xx')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy_xx')
                plt.imshow(pred_uy_xx,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy_xx))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy_xx')
                plt.imshow(gt_uy_xx_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_xx_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy_xx - pred_uy_xx)/max(gt_uy_xx)')
                plt.imshow(np.abs(gt_uy_xx_plot - pred_uy_xx)/np.max(np.abs(gt_uy_xx_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_xx_plot - pred_uy_xx)/np.max(np.abs(gt_uy_xx_plot)),interpolation='spline16'))
                plt.savefig('logs/uy_xx.png')

                gt_uy_yx_plot = gt_uy_yx[:FLATTEN_LEN].reshape(40,40) 

                plt.figure('uy_yx')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy_yx')
                plt.imshow(pred_uy_yx,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy_yx))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy_yx')
                plt.imshow(gt_uy_yx_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_yx_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy_yx - pred_uy_yx)/max(gt_uy_yx)')
                plt.imshow(np.abs(gt_uy_yx_plot - pred_uy_yx)/np.max(np.abs(gt_uy_yx_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_yx_plot - pred_uy_yx)/np.max(np.abs(gt_uy_yx_plot)),interpolation='spline16'))
                plt.savefig('logs/uy_yx.png')

                gt_uy_yy_plot = gt_uy_yy[:FLATTEN_LEN].reshape(40,40) 
                
                plt.figure('uy_yy')
                plt.subplot(1,3,1)
                plt.cla()
                plt.title('pred_uy_yy')
                plt.imshow(pred_uy_yy,interpolation='spline16')
                plt.colorbar(plt.imshow(pred_uy_yy))
                plt.subplot(1,3,2)
                plt.cla()
                plt.title('gt_uy_yy')
                plt.imshow(gt_uy_yy_plot,interpolation='spline16')
                plt.colorbar(plt.imshow(gt_uy_yy_plot))
                plt.subplot(1,3,3)
                plt.cla()
                plt.title('(gt_uy_yy - pred_uy_yy)/max(gt_uy_yy)')
                plt.imshow(np.abs(gt_uy_yy_plot - pred_uy_yy)/np.max(np.abs(gt_uy_yy_plot)),interpolation='spline16')
                plt.colorbar(plt.imshow(np.abs(gt_uy_yy_plot - pred_uy_yy)/np.max(np.abs(gt_uy_yy_plot)),interpolation='spline16'))
                plt.savefig('logs/uy_yy.png')

                
                
                
                
                


                plt.pause(0.01)
                
                

    
  


def main():
    global FLATTEN_LEN
    

    source_list = [3, 4, 5, 6, 7]
    source_num = len(source_list)

    layers = [3, 120, 120, 80, 2]


    data, x, y, s, gt_ux, gt_uy, gt_ux_xx, gt_ux_xy, gt_ux_yy, gt_uy_xx, gt_uy_yx, gt_uy_yy = PrepareData(data_path='heterogenous_nomalized_center_data_40by40_source34567.mat', source_list=source_list)

    FLATTEN_LEN = data['ux'].shape[1] * data['ux'].shape[2]

 
    X_nl = np.hstack((x, y, s))

    u_xy_nl = np.hstack((gt_ux, gt_uy))


 

    ####
    model = PhysicsInformedNN(X_nl, u_xy_nl, layers, source_num)    
    
    Epoch = 2000001
    model.train(nIter=Epoch,gt_ux=gt_ux, gt_ux_xx=gt_ux_xx, gt_ux_xy=gt_ux_xy, gt_ux_yy=gt_ux_yy, gt_uy=gt_uy, gt_uy_xx=gt_uy_xx, gt_uy_yx=gt_uy_yx, gt_uy_yy=gt_uy_yy, source_num=source_num)
    



if __name__ == '__main__':
    main()