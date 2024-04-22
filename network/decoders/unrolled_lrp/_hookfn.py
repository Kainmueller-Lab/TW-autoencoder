import torch
from torch import nn

class FwdHooks:
    def __init__(self,xai):
        self.xai=xai
        self.allowed_pass_layers = (torch.nn.ReLU, torch.nn.ELU, 
                           torch.nn.Dropout, torch.nn.Dropout2d,
                           torch.nn.Dropout3d,
                           torch.nn.Softmax,
                           torch.nn.LogSoftmax,
                           torch.nn.Sigmoid,
                           torch.nn.Tanh,
                           torch.nn.LeakyReLU)  

    def get_layer_fwd_hook(self,layer):
       

        if isinstance(layer,
                    (torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d)):
            return self.max_pool_nd_fwd_hook

        if isinstance(layer,
                    (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        
            return self.conv_nd_fwd_hook

        if isinstance(layer, torch.nn.Linear):
            return self.linear_fwd_hook
        
        
        if isinstance(layer,(torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                        torch.nn.BatchNorm3d)):
            return self.batch_norm_nd_fwd_hook

        if isinstance(layer, (torch.nn.AdaptiveAvgPool2d,torch.nn.AdaptiveAvgPool1d)):
        
            return self.avgpool_fwd_hook

        if isinstance(layer, self.allowed_pass_layers):

            return self.silent_pass # must return a function in hook_func form!
        else:
            raise NotImplementedError("The network contains layers that"
                                    " are currently not supported {0:s}".format(str(layer)))




    def max_pool_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):
        # Ignore unused for pylint
        _ = self

        # Save the return indices value to make sure
        tmp_return_indices = bool(m.return_indices)
        m.return_indices = True
        _, indices = m.forward(in_tensor[0])
        m.return_indices = tmp_return_indices
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "indices", indices)
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())

        return
    def avgpool_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):

        # contain output size

        stride=(in_tensor[0].shape[-2]//m.output_size[-2],in_tensor[0].shape[-1]//m.output_size[-1])

        k0 = in_tensor[0].size()[-2] - (m.output_size[-2]-1)*stride[-2]
        k1 = in_tensor[0].size()[-1] - (m.output_size[-1]-1)*stride[-1]
        kernel_size=(k0,k1)
        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'in_shape', in_tensor[0].size())
        setattr(m, 'stride', stride)
        setattr(m, 'kernel_size',kernel_size)
        setattr(m, 'in_channels',in_tensor[0].size()[1])


    def conv_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                         out_tensor: torch.Tensor):
        

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, 'out_shape', list(out_tensor.size()))
        return
    
    
    def linear_fwd_hook(self, m, in_tensor: torch.Tensor,
                        out_tensor: torch.Tensor):

        setattr(m, "in_tensor", in_tensor[0])
        setattr(m, "out_shape", list(out_tensor.size()))
        return
   
    def batch_norm_nd_fwd_hook(self, m, in_tensor: torch.Tensor,
                             out_tensor: torch.Tensor):   
        
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_tensor', in_tensor[0])
        setattr(m, 'out_tensor', out_tensor)

    def silent_pass(self, m, in_tensor: torch.Tensor,
                    out_tensor: torch.Tensor):
        #change by Xiaoyan
        pass
        # print("silent pass through ", m)
        
        # Placeholder forward hook for layers that do not need
        # to store any specific data. Still useful for module tracking.
        setattr(m, 'out_shape', out_tensor.size())
        setattr(m, 'in_shape', in_tensor[0].size())
