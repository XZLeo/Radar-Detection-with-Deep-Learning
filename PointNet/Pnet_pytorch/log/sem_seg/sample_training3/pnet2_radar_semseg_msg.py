'''
Configurable file for PointNet++ 
network in pytorch
'''
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()       
        
        self.sa1 = PointNetSetAbstractionMsg(1024, [1, 3], [8, 32], 4, 
                                             [[32, 32, 64], [64, 64, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [2, 4], [8, 32], 64+128, 
                                             [[32, 32, 64], [64, 64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [3, 6], [16, 32], 64+128, 
                                             [[64, 64, 128], [64, 64, 128]])
        
        self.fp1 = PointNetFeaturePropagation(256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(448, [128, 128])
        self.fp3 = PointNetFeaturePropagation(132, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 256, padding=127) 
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)        
        self.conv2 = nn.Conv1d(128, 128, 128, padding=63)
        self.bn2 = nn.BatchNorm1d(128) 
        self.drop2 = nn.Dropout(0.5)        
        self.conv3 = nn.Conv1d(128, num_classes, 6, padding=3)

    def forward(self, xyv):
        '''
        Forward pass (traversing through all neurons from first to last layer)
        Input: xyv = x,y positions and v radial velocity
        Output: x = xy feature points; l3_points = xyv feature points
        '''
        l0_points = xyv
        l0_xy = xyv[:,:2,:]
                
        l1_xy, l1_points = self.sa1(l0_xy, l0_points)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)
        
        #self.fp(nl-1, nl, nl-1_p, nl_p)  
        l2_points = self.fp1(l2_xy, l3_xy, None, l3_points) 
        l1_points = self.fp2(l1_xy, l2_xy, l1_points, l2_points)
        l0_points = self.fp3(l0_xy, l1_xy, l0_points, l1_points)     
        
        l0_points_pad = F.pad(input=l0_points, pad=(1, 0, 0, 0), 
                              mode='constant', value=0) #add unitary column of zeros to 3097 points
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points_pad)))) 
        x = self.drop2(F.relu(self.bn2(self.conv2(x)))) 
        x = self.conv3(x)
                
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()    
        
    def forward(self, pred, target, trans_feat, weight):
        #total_loss = F.nll_loss(pred, target, weight=weight)
        total_loss = F.cross_entropy(pred, target, weight=weight)
        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))