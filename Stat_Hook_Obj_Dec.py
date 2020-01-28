import torch
import numpy as np
import torchvision.models.detection as detection
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch import nn
from collections import OrderedDict
import torchvision
class statistics():
    def __init__(self, module):                                                
        self.hook = module.register_forward_hook(self.hook_fn)
        self.min_input = np.inf
        self.max_input = - np.inf
        self.min_output = np.inf
        self.max_output = -np.inf
        self.sum_min_input = 0
        self.sum_max_input = 0
        self.sum_min_output = 0
        self.sum_max_output = 0                           
        self.avg_min_input = 0                  
        self.avg_max_input = 0          
        self.avg_min_output = 0                                   
        self.avg_max_output = 0 
        self.count_input = 0
        self.count_output = 0

    def hook_fn(self,module,input,output):  
        if (not isinstance(module,torchvision.ops.feature_pyramid_network.LastLevelMaxPool)) and (not isinstance(module,torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork)) and (not isinstance(module,torchvision.models.detection.backbone_utils.BackboneWithFPN)) and (not isinstance(module,GeneralizedRCNNTransform))and (not isinstance(module,torchvision.models.detection.roi_heads.RoIHeads))and (not isinstance(module,torchvision.models.detection.mask_rcnn.MaskRCNN)):
        
            if type(input) == tuple or type(input) == list:                   
                for i in range(len(input)): 
                    if (input[i]) is not None and (type(input[i]) == list or type(input[i])==torch.Tensor):  
                        for j in range(len(input[i])):
                            if type(input[i][j])== torch.Tensor :
                                if torch.min(input[i][j]) < self.min_input:  
                                    self.min_input = torch.min(input[i][j])      
                                if torch.max(input[i][j]) > self.max_input: 
                                    self.max_input = torch.max(input[i][j])
                        
                                self.sum_min_input += torch.min(input[i][j])  
                                self.sum_max_input += torch.max(input[i][j])
                                self.count_input += 1
                        self.avg_min_input = self.sum_min_input / self.count_input
                        self.avg_max_input =  self.sum_max_input / self.count_input

            if type(output) == torch.Tensor:
                if torch.min(output) < self.min_output:
                    self.min_output = torch.min(output)
                if torch.max(output) > self.max_output:
                    self.max_output = torch.max(output)
                for b in range(len(output)):
                    self.sum_min_output += torch.min(output[b])     
                    self.sum_max_output += torch.max(output[b])
                    self.count_output += 1
                self.avg_min_input = self.sum_min_input / self.count_output
                self.avg_max_input =  self.sum_max_input / self.count_output
                self.avg_min_output = self.sum_min_output /self.count_output
                self.avg_max_output = self.sum_max_output / self.count_output
            
            elif isinstance(output,dict): 
                for i in range(len(output)):
                    if torch.min(output[i]) < self.min_output:
                        self.min_output = torch.min(output[i])
                    if torch.max(output[i]) > self.max_output:
                        self.max_output = torch.max(output[i])

                    for j in range(len(output[i])):
                        self.sum_min_output += torch.min(output[i][j])     
                        self.sum_max_output += torch.max(output[i][j])
                        self.count_output += 1
                    self.avg_min_input = self.sum_min_input / self.count_output
                    self.avg_max_input =  self.sum_max_input / self.count_output
                    self.avg_min_output = self.sum_min_output /self.count_output
                    self.avg_max_output = self.sum_max_output / self.count_output



            elif isinstance(output,tuple):
                for i in range(len(output)):                
                    for j in range(len(output[i])):
                        if torch.min(output[i][j]) < self.min_output:
                            self.min_output = torch.min(output[i][j])
                        if torch.max(output[i][j]) > self.max_output:
                            self.max_output = torch.max(output[i][j])

            elif isinstance(output,list):     
                for i in range(len(output)):
                    if torch.min(output[i]) < self.min_output:
                        self.min_output = torch.min(output[i])
                    if torch.max(output[i]) > self.max_output:
                        self.max_output = torch.max(output[i])
    def close(self):
        self.hook.remove()