import os
import numpy as np
import torch
import pickle

class ClassicalPLMCLSCalibrator:
    def __init__(self, args) -> None:
        self.args = args
        self.y_dict= {'y_logits_dev' : torch.empty(0).cpu(),
        'y_dev' : torch.empty(0).cpu(),
        'y_logits' : torch.empty(0).cpu(),
        'y_true' : torch.empty(0).cpu()}


    def cat_tensor_to_variable(self,tensor_,name):
        if name in self.y_dict.keys():
            self.y_dict[name] = torch.cat((self.y_dict[name], tensor_.cpu()), dim=0)

    def write_pickled_data_to_file(self):
        # Write file with pickled data
        with open(self.args.logits_dir, 'wb') as f:
            pickle.dump([(self.y_dict['y_logits_dev'].cpu().numpy(), self.y_dict['y_dev'].cpu().numpy().astype(int)[:,np.newaxis]),
                (self.y_dict['y_logits'].cpu().numpy(), self.y_dict['y_true'].cpu().numpy().astype(int)[:,np.newaxis])], f)