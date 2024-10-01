# import numpy as np
from torch.utils.data import Dataset



class navier_lame_dataset(Dataset):
    """docstring for navier_lame_dataset."""

    def __init__(self, ux, uy, ux_xx, ux_xy, ux_yy, uy_xx, uy_yx, uy_yy, x, y):
        super(navier_lame_dataset, self).__init__()
        self.ux = ux
        self.uy = uy
        self.ux_xx = ux_xx
        self.ux_xy = ux_xy
        self.ux_yy = ux_yy
        self.uy_xx = uy_xx
        self.uy_yx = uy_yx
        self.uy_yy = uy_yy  
        self.x = x  
        self.y = y  


    def __len__(self):
        return len(self.ux)

    def __getitem__(self, index):
        return self.ux[index], self.uy[index], self.ux_xx[index], self.ux_xy[index], self.ux_yy[index], self.uy_xx[index], self.uy_yx[index], self.uy_yy[index], self.x[index], self.y[index]




