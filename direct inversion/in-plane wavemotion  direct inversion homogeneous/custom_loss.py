import torch
import torch.nn as nn
from pdb import set_trace as st
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pinn_loss (ux, uy, ux_xx, ux_xy, ux_yy, uy_xx, uy_yx, uy_yy, output1, output2, output3):
    w = 3.91
    rho_parameter = 1
    mu_parameter = output1.to(device)
    lambda_parameter = output2.to(device)
    delta_parameter =  output3.to(device)

    c1 = ((w**2) * ux) + (mu_parameter * (ux_xx + ux_yy) / rho_parameter) + ((lambda_parameter+mu_parameter)*(ux_xx + uy_yx) / rho_parameter)
    c2 = ((w**2) * uy) + (mu_parameter * (uy_xx + uy_yy) / rho_parameter) + ((lambda_parameter+mu_parameter)*(ux_xy + uy_yy) / rho_parameter)


    c1_loss = torch.mean(c1**2)
    c2_loss = torch.mean(c2**2)


    loss = c1_loss + c2_loss + (torch.mean(delta_parameter*(mu_parameter)**2)+torch.mean(delta_parameter*(lambda_parameter)**2))
    


    return loss, c1_loss, c2_loss




