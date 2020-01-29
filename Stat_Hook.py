import torch
import numpy as np

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
        self.count = 0
        
    def hook_fn(self, module, input, output):     type(input) 
        
        if torch.min(input[0]) < self.min_input:
            self.min_input = torch.min(input[0])

        if torch.max(input[0]) > self.max_input:
            self.max_input = torch.max(input[0])

        if torch.min(output) < self.min_output:
            self.min_output = torch.min(output)
        if torch.max(output) > self.max_output:
            self.max_output = torch.max(output)
        for b in range(input[0].shape[0]):
            self.sum_min_input += torch.min(input[0][b])
            self.sum_max_input += torch.max(input[0][b])
            self.sum_min_output += torch.min(output[b])
            self.sum_max_output += torch.max(output[b])
            self.count += 1
        self.avg_min_input = self.sum_min_input / self.count
        self.avg_max_input =  self.sum_max_input / self.count
        self.avg_min_output = self.sum_min_output / self.count
        self.avg_max_output = self.sum_max_output / self.count

        

    def close(self):
        self.hook.remove()