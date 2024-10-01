import numpy as np 
from pdb import set_trace as st
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from model import UNet, FCN
from data_gen import navier_lame_dataset


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader


from custom_loss import pinn_loss

import scipy.io
import pandas as pd
#import wandb

#wandb.init(project="my-test-project", entity="yaxu2271")



def main():
    MODE = 'eval' # 'train', 'eval', 'test'

    ### training network
    if not os.path.exists('./model/'):
        os.mkdir('./model/')

    if not os.path.exists('./result/'):
        os.mkdir('./result/')

    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')

    ### Hyperparameters
    BATCH_SIZE = 8*1600
    EPOCH = 10000
    LEARNING_RATE = 5e-5
    
    input_size_NN =1600*2
    predefine_weights_bias = False
    weights_bias = [True,[np.zeros((64,1600*2)),np.zeros((1600,64))]]

    # st()

    print('Preparing dataset:')
    # data = scipy.io.loadmat('../data/FGrad.mat')
    data = scipy.io.loadmat('homogeneous_nomalized_center_data_40by40_source3.mat')
    list_of_keys = list(data.keys())
    list_of_keys.remove('__header__')
    list_of_keys.remove('__version__')
    list_of_keys.remove('__globals__')
  
    
    
    for each_key in list_of_keys:
        # vars()[each_key] = data[each_key].flatten()
        vars()[each_key] = torch.from_numpy(np.array(data[each_key].flatten())).type(torch.FloatTensor)
    


    print('Initializing Network...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Now using device {device}!')


    
    # using_model1 = nn.DataParallel(FCN(input_size=input_size_NN)).to(device)
    # using_model2 = nn.DataParallel(FCN(input_size=input_size_NN)).to(device)
    # using_model3 = nn.DataParallel(FCN(input_size=input_size_NN)).to(device)

    if predefine_weights_bias:

        using_model1 = FCN(input_size=input_size_NN,weights_bias=weights_bias).to(device)
        using_model2 = FCN(input_size=input_size_NN,weights_bias=weights_bias).to(device)
        using_model3 = FCN(input_size=input_size_NN,weights_bias=weights_bias).to(device)

    else:

        using_model1 = FCN(input_size=input_size_NN).to(device)
        using_model2 = FCN(input_size=input_size_NN).to(device)
        using_model3 = FCN(input_size=input_size_NN).to(device)

    
    train_data = navier_lame_dataset(vars()['ux'], 
                                     vars()['uy'], 
                                     vars()['ux_xx'], 
                                     vars()['ux_xy'],
                                     vars()['ux_yy'], 
                                     vars()['uy_xx'], 
                                     vars()['uy_yx'], 
                                     vars()['uy_yy'],
                                     vars()['x'],
                                     vars()['y'])

    data_loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    




    # criterion = nn.MSELoss()

    if MODE == 'train':


        print("Training Mode")
        optimizer = optim.Adam(list(using_model1.parameters())+list(using_model2.parameters())+list(using_model3.parameters()), lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-6)

        lambda_list = []
        mu_list = []
        c1_loss_list = []
        c2_loss_list = []
        loss_list = []
        delta_list = []
        
        
        
        
        # l_1 = np.concatenate((np.ones((20,20))*(6/3),np.ones((20,20))*(8/3)),1)
        # l_2 = np.concatenate((np.ones((20,20))*(2/3),np.ones((20,20))*(4/3)),1)
        # lambda_gt = np.concatenate((l_1,l_2),0)

        # m_1 = np.concatenate((np.ones((20,20))*(3),np.ones((20,20))*(4)),1)
        # m_2 = np.concatenate((np.ones((20,20))*(1),np.ones((20,20))*(2)),1)
        # mu_gt = np.concatenate((m_1,m_2),0)

        # z_gt = np.concatenate((lambda_gt.flatten()[np.newaxis,:], mu_gt.flatten()[np.newaxis,:]),0)
                
        # z_tensor = Variable(torch.from_numpy(z_gt), requires_grad=True)

        # st()
        # np.ones((20,20))*(6/3)
        # np.ones((20,20))*(8/3)
        # np.ones((20,20))*(2/3)
        # np.ones((20,20))*(4/3)
        
        # z = torch.ones(1, input_size, requires_grad=True)
        loss_m = 999
        plt.ion()
        for epoch in range(1,EPOCH+1):

            for each_ux, each_uy, each_ux_xx, each_ux_xy, each_ux_yy, each_uy_xx, each_uy_yx, each_uy_yy, each_x, each_y in data_loader_train:
                input_ = torch.cat([each_x, each_y]).to(device)
                each_ux = each_ux.to(device)
                each_uy = each_uy.to(device)
                each_ux_xx = each_ux_xx.to(device)
                each_ux_xy = each_ux_xy.to(device)
                each_ux_yy = each_ux_yy.to(device)
                each_uy_xx = each_uy_xx.to(device)
                each_uy_yx = each_uy_yx.to(device)
                each_uy_yy = each_uy_yy.to(device)
                # st()

                optimizer.zero_grad()
                
                output1 = using_model1(input_).reshape((1,1600))
                output2 = using_model2(input_).reshape((1,1600))
                output3 = using_model3(input_).reshape((1,1600))
                # st()

                # output = Variable(torch.tensor(0.47), requires_grad=True)

                loss, c1_loss, c2_loss = pinn_loss(each_ux, 
                                                each_uy, 
                                                each_ux_xx, 
                                                each_ux_xy, 
                                                each_ux_yy, 
                                                each_uy_xx, 
                                                each_uy_yx, 
                                                each_uy_yy, 
                                                #z_tensor,#
                                                output1,
                                                output2,
                                                output3)

                # st()
                
                loss.backward()
                optimizer.step()


                
                if epoch > EPOCH-2 :
                    loss_m = loss.item()
                    print(loss_m)
                    mu_ = output1.detach().cpu().numpy().reshape((40,40))
                    lam_ = output2.detach().cpu().numpy().reshape((40,40))
                    delta_ = output3.detach().cpu().numpy().reshape((40,40))
                    print('-----------------------------------------------')
                    print(f'Mu(1): {np.mean(mu_[:20,:20])}')
                    print(f'Mu(2): {np.mean(mu_[:20,20:40])}')
                    print(f'Mu(3): {np.mean(mu_[20:40,:20])}')
                    print(f'Mu(4): {np.mean(mu_[20:40,20:40])}')

                    print(f'Lam(1): {np.mean(lam_[:20,:20])}')
                    print(f'Lam(2): {np.mean(lam_[:20,20:40])}')
                    print(f'Lam(3): {np.mean(lam_[20:40,:20])}')
                    print(f'Lam(4): {np.mean(lam_[20:40,20:40])}')
                    
                    print(f'Delta(1): {np.mean(delta_[:20,:20])}')
                    print(f'Delta(2): {np.mean(delta_[:20,20:40])}')
                    print(f'Delta(3): {np.mean(delta_[20:40,:20])}')
                    print(f'Delta(4): {np.mean(delta_[20:40,20:40])}')
                    print('-----------------------------------------------')
                    
                    matplotlib.pyplot.close('all')
                    plt.figure('Parameters')
                    #plt.subplot(1,2,1)
                    plt.cla()
                    plt.title('Mu')
                    plt.imshow(mu_,interpolation='spline16', cmap='Blues')
                    plt.colorbar(plt.imshow(mu_,interpolation='spline16', cmap='Blues'))
                    #plt.subplot(1,2,2)
                    plt.cla()
                    plt.title('Lam')
                    plt.imshow(lam_,interpolation='spline16', cmap='Blues')
                    plt.colorbar(plt.imshow(lam_,interpolation='spline16', cmap='Blues'))
                    plt.figure('Delta')
                    plt.cla()
                    plt.title('Delta')
                    plt.imshow(delta_,interpolation='spline16', cmap='Blues')
                    plt.colorbar(plt.imshow(delta_,interpolation='spline16', cmap='Blues'))

                    plt.pause(0.01)

                # wandb.log()
                
            # wandb.log()
            
                print(f'-------------Epoch {epoch}---------------')
                # print(f'mu: {output[0,0].detach().numpy()}, lambda: {output[0,1].detach().numpy()}')
                print(f'Loss:{loss.detach().cpu().numpy()}, c1 loss:{c1_loss.detach().cpu().numpy()}, c2 loss:{c2_loss.detach().cpu().numpy()}')

            lambda_list.append(output2.detach().cpu().numpy())
            mu_list.append(output1.detach().cpu().numpy())
            delta_list.append(output3.detach().cpu().numpy())
            c1_loss_list.append(c1_loss.detach().cpu().numpy())
            c2_loss_list.append(c2_loss.detach().cpu().numpy())
            loss_list.append(loss.detach().cpu().numpy())
            
            if epoch == 1 or epoch%1000 == 0 or epoch == EPOCH:
                torch.save(using_model1.state_dict(),'./model/parameter_estimation_navier_lame_mu_{}.pth'.format(epoch))
                torch.save(using_model2.state_dict(),'./model/parameter_estimation_navier_lame_lambda_{}.pth'.format(epoch))
                torch.save(using_model3.state_dict(),'./model/parameter_estimation_navier_lame_delta_{}.pth'.format(epoch))

        plt.figure('Parameters1')
        #plt.subplot(1,2,1)
        plt.title('Lambda')
        c_lambda = plt.imshow(lambda_list[-1].reshape((40,40)),interpolation='spline16', cmap='Blues')
        plt.colorbar(c_lambda)
        plt.savefig('./logs/lambda_prediction_distribution.png')
        # plt.plot(lambda_list)
        #plt.subplot(1,2,2)
        plt.figure('Parameters2')
        plt.title('Mu')
        c_mu = plt.imshow(mu_list[-1].reshape((40,40)),interpolation='spline16', cmap='Blues')
        # plt.plot(mu_list)
        plt.colorbar(c_mu)
        plt.savefig('./logs/lmu_prediction_distribution.png')
        plt.figure('Error_Corrector')
        #plt.subplot(1,2,1)
        plt.title('Delta')
        c_delta = plt.imshow(delta_list[-1].reshape((40,40)),interpolation='spline16', cmap='Blues')
        #plt.plot(mu_list)
        plt.colorbar(c_delta)
        plt.savefig('./logs/delta_distribution.png')
        
        np.save('./result/lambda',lambda_list[-1].reshape((40,40)))
        np.save('./result/mu',mu_list[-1].reshape((40,40)))
        np.save('./result/delta',delta_list[-1].reshape((40,40)))
        lambda_prediction = pd.DataFrame(lambda_list[-1].reshape((40,40)))
        mu_prediction = pd.DataFrame(mu_list[-1].reshape((40,40)))
        delta_prediction = pd.DataFrame(delta_list[-1].reshape((40,40)))

        lambda_prediction.to_csv("lambda_prediction.csv")
        mu_prediction.to_csv("mu_prediction.csv")
        delta_prediction.to_csv("delta_prediction.csv")


        plt.figure('Loss')
        plt.subplot(1,2,1)
        plt.title('Weighted Loss')
        plt.plot(np.log10(loss_list))
        plt.subplot(1,2,2)
        plt.title('Sub Losses')
        plt.plot(c1_loss_list,'r-.', label='C1 Loss')
        plt.plot(c2_loss_list,'b--',label='C2 Loss')
        plt.legend()
        plt.savefig('./logs/loss.png')
        plt.show()


        loss_list_prediction = pd.DataFrame(loss_list)
        loss_list_prediction.to_csv("./logs/loss_list_prediction.csv")
        
        
        plt.figure('Total Loss')
        plt.title('Total Loss')
        plt.plot(np.log10(loss_list))
        plt.savefig('./logs/total_loss.png')
        plt.show()
        # st()

    if MODE == 'test':
         MODEL_1_PATH = './model/parameter_estimation_navier_lame_mu_10000.pth'
         MODEL_2_PATH = './model/parameter_estimation_navier_lame_lambda_10000.pth'
         MODEL_3_PATH = './model/parameter_estimation_navier_lame_delta_10000.pth'

         using_model1.load_state_dict(torch.load(MODEL_1_PATH,map_location=torch.device('cpu')))
         using_model2.load_state_dict(torch.load(MODEL_2_PATH,map_location=torch.device('cpu')))
         using_model3.load_state_dict(torch.load(MODEL_3_PATH,map_location=torch.device('cpu')))

         using_model1.eval()
         using_model2.eval()
         using_model3.eval()
        
         for name, param in using_model1.named_parameters():
             print(name, param.shape)
             cur_pd = pd.DataFrame(param.detach().cpu().numpy())
            
             cur_pd.to_csv(f'./result/mu/{name}.csv')
             
         for name, param in using_model2.named_parameters():
             print(name, param.shape)
             cur_pd = pd.DataFrame(param.detach().cpu().numpy())
           
             cur_pd.to_csv(f'./result/lambda/{name}.csv')
             
         for name, param in using_model3.named_parameters():
             print(name, param.shape)
             cur_pd = pd.DataFrame(param.detach().cpu().numpy())
          
             cur_pd.to_csv(f'./result/delta/{name}.csv')
            
    if MODE == 'eval':
        MODEL_1_PATH = './model/parameter_estimation_navier_lame_mu_10000.pth'
        MODEL_2_PATH = './model/parameter_estimation_navier_lame_lambda_10000.pth'
        MODEL_3_PATH = './model/parameter_estimation_navier_lame_delta_10000.pth'

        using_model1.load_state_dict(torch.load(MODEL_1_PATH,map_location=torch.device('cpu')))
        using_model2.load_state_dict(torch.load(MODEL_2_PATH,map_location=torch.device('cpu')))
        using_model3.load_state_dict(torch.load(MODEL_3_PATH,map_location=torch.device('cpu')))

        using_model1.eval()
        using_model2.eval()
        using_model3.eval()
        
        for each_ux, each_uy, each_ux_xx, each_ux_xy, each_ux_yy, each_uy_xx, each_uy_yx, each_uy_yy, each_x, each_y in data_loader_train:
            input_ = torch.cat([each_x, each_y]).to(device)

            output1 = using_model1(input_).reshape((1,1600))
            output2 = using_model2(input_).reshape((1,1600))
            output3 = using_model3(input_).reshape((1,1600))
            
            mu_ = output1.detach().cpu().numpy().reshape((40,40))
            lam_ = output2.detach().cpu().numpy().reshape((40,40))
            delta_ = output3.detach().cpu().numpy().reshape((40,40))
            
            print('print mean-----------------------------------------------')
            print(f'Mu: {np.mean(mu_[:,:])}')

            print(f'Lam: {np.mean(lam_[:,:])}')
            
            
            print(f'Delta: {np.mean(delta_[:,:])}')
            
            
            
            print('print std-----------------------------------------------')
            print(f'Mu: {np.std(mu_[:,:])}')
            

            print(f'Lam: {np.std(lam_[:,:])}')
            
            
            print(f'Delta: {np.std(delta_[:,:])}')
            
            print('-----------------------------------------------')

            mu_eval = output1.detach().cpu().numpy()
            average_mu_eval = np.average(mu_eval)
            print('average_mu_eval:')
            print(average_mu_eval)
            max_mu_residual = np.max(np.abs(mu_eval-1))
            print('max_mu_residual:')
            print(max_mu_residual)
            min_mu_residual = np.min(np.abs(mu_eval-1))
            print('min_mu_residual:')
            print(min_mu_residual)
            lambda_eval = output2.detach().cpu().numpy()
            average_lambda_eval = np.average(lambda_eval)
            print('average_lambda_eval:')
            print(average_lambda_eval)
            max_lambda_residual = np.max(np.abs(lambda_eval-0.47))
            min_lambda_residual = np.min(np.abs(lambda_eval-0.47))
            print('max_lambda_residual:')
            print(max_lambda_residual)
            print('min_lambda_residual:')
            print(min_lambda_residual)
            delta_eval = output3.detach().cpu().numpy()


            plt.figure('Lambda')
            plt.title('Lambda')
            c_lambda = plt.imshow(lambda_eval.reshape((40,40)),interpolation='spline16', cmap='Blues')
            plt.colorbar(c_lambda)
            plt.clim(0.3,0.7)
            plt.savefig('./logs/lambda_prediction_distribution.png')

            
            plt.figure('Misfit of lambda')
            plt.title('Misfit of lambda')
            c_lambda_misfit = plt.imshow(np.abs(lambda_eval.reshape((40,40))-0.47)/0.47,interpolation='spline16', cmap='Blues')
            plt.colorbar(c_lambda_misfit)
            plt.clim(0,0.2)
            plt.savefig('./logs/lambda_misfit_distribution.png')


            plt.figure('Ground truth of lambda')
            plt.title('Ground truth of lambda')
            c_lambda_gt = plt.imshow(lambda_eval.reshape((40,40))-lambda_eval.reshape((40,40))+0.47,interpolation='spline16', cmap='Blues')
            plt.colorbar(c_lambda_gt)
            plt.savefig('./logs/lambda_gt_distribution.png')

            
            plt.figure('Mu')
            plt.title('Mu')
            c_mu = plt.imshow(mu_eval.reshape((40,40)),interpolation='spline16', cmap='Blues')
            plt.colorbar(c_mu)
            plt.clim(0.8,1.2)
            plt.savefig('./logs/mu_prediction_distribution.png')

            
            plt.figure('Misfit of mu')
            plt.title('Misfit of mu')
            c_mu_misfit = plt.imshow(np.abs(mu_eval.reshape((40,40))-1),interpolation='spline16', cmap='Blues')
            plt.colorbar(c_mu_misfit)
            plt.clim(0,0.2)
            plt.savefig('./logs/mu_misfit_distribution.png')


            plt.figure('Ground truth of mu')
            plt.title('Ground truth of mu')
            c_mu_gt = plt.imshow(mu_eval.reshape((40,40))-mu_eval.reshape((40,40))+1,interpolation='spline16', cmap='Blues')
            plt.colorbar(c_mu_gt)
            plt.savefig('./logs/mu_gt_distribution.png')

            
            plt.figure('Error_Corrector')
            plt.title('Delta')
            c_delta = plt.imshow(delta_eval.reshape((40,40)),interpolation='spline16', cmap='Blues')
            plt.colorbar(c_delta)
            plt.savefig('./logs/delta_distribution.png')

            plt.show()
            
            plt.figure('Loss')
            plt.subplot(1,2,1)
            plt.title('Weighted Loss')
            plt.plot(loss_list)
            plt.subplot(1,2,2)
            plt.title('Sub Losses')
            plt.plot(c1_loss_list,'.-', label='C1 Loss')
            plt.plot(c2_loss_list,'--',label='C2 Loss')
            plt.legend()
            plt.savefig('./logs/loss.png')
            plt.show()
            # break
            # st()

        # for name, param in using_model1.named_parameters():
        #     print(name, param.shape)
        #     cur_pd = pd.DataFrame(param.detach().cpu().numpy())
            
        #     cur_pd.to_csv(f'./result/{name}.csv')

if __name__ == '__main__':
    main()