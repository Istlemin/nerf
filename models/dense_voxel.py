from sys import int_info
import torch
from torch import nn
import torch.nn.functional as F

class DenseVoxel(nn.Module):
    def __init__(self, res=20, device="cpu"):
        super(DenseVoxel, self).__init__()

        self.res = res
        self.start_coord = -2
        self.end_coord = 2

        self.color_grid = torch.nn.Parameter(torch.ones((res,res,res,3)))
        self.density_grid = torch.nn.Parameter(torch.ones((res,res,res)))
    
    def forward(self, points, dirs, device="cpu"):
        float_indices = (points-self.start_coord)/(self.end_coord-self.start_coord)*self.res
        int_indices = (float_indices).type(torch.cuda.LongTensor)
        int_indices = torch.maximum(torch.tensor(0,device=device),int_indices)
        int_indices = torch.minimum(torch.tensor(self.res-1,device=device),int_indices)

        colors = self.color_grid[int_indices[:,0],int_indices[:,1],int_indices[:,2]]
        densities = self.density_grid[int_indices[:,0],int_indices[:,1],int_indices[:,2]]
        densities *= (points.min(dim=-1)[0]>=self.start_coord) & (points.max(dim=-1)[0]<=self.end_coord)
    
        return torch.sigmoid(colors), torch.sigmoid(densities)