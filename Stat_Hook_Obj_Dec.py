import torch
import numpy as np

class tensor_stat():
    def __init__(self):
        self.min = np.inf
        self.max = - np.inf
        self.sum_min = 0
        self.sum_max = 0                           
        self.avg_min = 0                  
        self.avg_max = 0          
        self.count = 0

    def update_stat(self, tensor):
        if torch.min(tensor) < self.min:
            self.min = torch.min(tensor)
        if torch.max(tensor) > self.max:
            self.max = torch.max(tensor)
        for b in range(len(tensor)):
            self.sum_min += torch.min(tensor[b])     
            self.sum_max += torch.max(tensor[b])
            self.count += 1
        self.avg_min = self.sum_min /self.count
        self.avg_max = self.sum_max / self.count

class statistics():
    def __init__(self, module):                                                
        self.hook = module.register_forward_hook(self.hook_fn)
        self.input_stat = tensor_stat()
        self.output_stat = tensor_stat()

    def hook_fn(self,module,input,output):  
        
        
        if type(input) == tuple or type(input) == list:                   
            for i in range(len(input)): 
                if type(input[i])==torch.Tensor:  
                    self.input_stat.update_stat(input[i])
        
        if type(output) == torch.Tensor:
            self.output_stat.update_stat(output)

    def close(self):
        self.hook.remove()