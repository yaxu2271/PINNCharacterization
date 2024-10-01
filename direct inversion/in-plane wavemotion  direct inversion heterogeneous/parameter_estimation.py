import numpy as np 
from pdb import set_trace as st
import matplotlib.pyplot as plt
import matplotlib
from model import FCN
from data_gen import navier_lame_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_loss import pinn_loss
import scipy.io
import pandas as pd
from tqdm import tqdm 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def main():
    MODE = 'train' # 'train', 'eval', 'test'

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



    print('Preparing dataset:')
    data = scipy.io.loadmat('heterogenous_nomalized_center_data_40by40_source3.mat')
    list_of_keys = list(data.keys())
    list_of_keys.remove('__header__')
    list_of_keys.remove('__version__')
    list_of_keys.remove('__globals__')
  
    
    
    for each_key in list_of_keys:
        vars()[each_key] = torch.from_numpy(np.array(data[each_key].flatten())).type(torch.FloatTensor)
    


    print('Initializing Network...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Now using device {device}!')


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
    





    if MODE == 'train':


        print("Training Mode")
        optimizer = optim.Adam(list(using_model1.parameters())+list(using_model2.parameters())+list(using_model3.parameters()), lr=LEARNING_RATE, betas=(0.9,0.999),eps=1e-6)

        lambda_list = []
        mu_list = []
        c1_loss_list = []
        c2_loss_list = []
        loss_list = []
        delta_list = []
        
        

        plt.ion()
        for epoch in tqdm(range(1,EPOCH+1)):

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


                loss, c1_loss, c2_loss = pinn_loss(each_ux, 
                                                each_uy, 
                                                each_ux_xx, 
                                                each_ux_xy, 
                                                each_ux_yy, 
                                                each_uy_xx, 
                                                each_uy_yx, 
                                                each_uy_yy, 
                                                output1,
                                                output2,
                                                output3)

                
                loss.backward()
                optimizer.step()


                
                if epoch > EPOCH-3 :
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
                    plt.subplot(1,2,1)
                    plt.cla()
                    plt.title('Mu')
                    plt.imshow(mu_)
                    plt.colorbar(plt.imshow(mu_))
                    plt.subplot(1,2,2)
                    plt.cla()
                    plt.title('Lam')
                    plt.imshow(lam_)
                    plt.colorbar(plt.imshow(lam_))
                    plt.figure('Delta')
                    plt.cla()
                    plt.title('Delta')
                    plt.imshow(delta_)
                    plt.colorbar(plt.imshow(delta_))

                    plt.pause(0.01)

                
            
                print(f'-------------Epoch {epoch}---------------')
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
        c_lambda = plt.imshow(lambda_list[-1].reshape((40,40)), cmap='Blues')
        plt.colorbar(c_lambda)
        plt.savefig('./logs/lambda_prediction_distribution.png')
        # plt.plot(lambda_list)
        #plt.subplot(1,2,2)
        plt.figure('Parameters2')
        plt.title('Mu')
        c_mu = plt.imshow(mu_list[-1].reshape((40,40)), cmap='Blues')
        # plt.plot(mu_list)
        plt.colorbar(c_mu)
        plt.savefig('./logs/lmu_prediction_distribution.png')
        plt.figure('Error_Corrector')
        #plt.subplot(1,2,1)
        plt.title('Delta')
        c_delta = plt.imshow(delta_list[-1].reshape((40,40)), cmap='Blues')
        #plt.plot(mu_list)
        plt.colorbar(c_delta)
        plt.savefig('./logs/delta_distribution.png')
        
        np.save('./result/lambda',lambda_list[-1].reshape((40,40)))
        np.save('./result/mu',mu_list[-1].reshape((40,40)))
        lambda_prediction = pd.DataFrame(lambda_list[-1].reshape((40,40)))
        mu_prediction = pd.DataFrame(mu_list[-1].reshape((40,40)))
        lambda_prediction.to_csv("lambda_prediction.csv")
        mu_prediction.to_csv("mu_prediction.csv")

        plt.figure('Loss')
        plt.subplot(1,2,1)
        plt.title('Weighted Loss')
        plt.plot(np.log10(loss_list))
        plt.subplot(1,2,2)
        plt.title('Sub Losses')
        plt.plot(np.log10(c1_loss_list), label='C1 Loss')
        plt.plot(np.log10(c2_loss_list),label='C2 Loss')
        plt.legend()
        plt.savefig('./logs/loss.png')
        plt.show()

        loss_list_prediction = pd.DataFrame(loss_list)
        loss_list_prediction.to_csv("./logs/loss_list_prediction.csv")



if __name__ == '__main__':
    main()