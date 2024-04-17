import torch
from torch import nn
from torch.nn import functional as F
from math import prod
# --------------------------------------
# Summary for LRP layres for all kinds of rules 
# --------------------------------------
'''
Summary for the LRP layer behavior under different rules:
    LRP epsilon:
        only activation, we do silent pass
    LRP alpha beta rule:
        also silent rule for batchnorm layer #TODO check
        use z bounidng rule for the first conv layer

'''

# --------------------------------------
# Basic class and utils function
# --------------------------------------


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())



class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R


    def f_linear(self,w1, w2, x1, x2, R):

        Z1 = F.linear(x1, w1)
        Z2 = F.linear(x2, w2)
        S1 = safe_divide(R, Z1)
        S2 = safe_divide(R, Z2)
        C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        C2 = x2 * self.gradprop(Z2, x2, S2)[0]

        return C1 + C2
    def f_conv2d(self,w1, w2, x1, x2, R, stride, padding):

        Z1 = F.conv2d(x1, w1, bias=None, stride=stride, padding=padding)
        Z2 = F.conv2d(x2, w2, bias=None, stride=stride, padding=padding)
        S1 = safe_divide(R, Z1)
        S2 = safe_divide(R, Z2)
        C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        C2 = x2 * self.gradprop(Z2, x2, S2)[0]

        return C1 + C2

# --------------------------------------
# Layers for LRP
# --------------------------------------
class LRP_linear(RelProp):
    def __init__(self, linear, **kwargs):
        super(LRP_linear, self).__init__()
        assert isinstance(linear,nn.Linear), "Please tie with a linear layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']

        self.m=linear       
        self.inv_m=nn.Linear(in_features=self.m.out_features, out_features=self.m.in_features, bias=None) #bias=None
        self.inv_m.weight=nn.Parameter(self.m.weight.t())

    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out


    def LRP_eps_forward(self,relevance_in):
        relevance_out = self.inv_m(safe_divide(relevance_in,self.m.out_tensor+self.eps*torch.mean(self.m.in_tensor)))
     
        #relevance_out = self.inv_m(relevance_in/(self.m.out_tensor+self.eps*torch.mean(self.m.in_tensor)))
        relevance_out *= self.m.in_tensor
  

        assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"
        return relevance_out

    def LRP_ab_forward(self,relevance_in):
        beta=self.alpha-1
        pw = torch.clamp(self.m.weight, min=0)
        nw = torch.clamp(self.m.weight, max=0)
        px = torch.clamp(self.m.in_tensor, min=0)
        nx = torch.clamp(self.m.in_tensor, max=0)
     
        relevance_out = self.inv_m(relevance_in/(self.m.out_tensor+self.eps))
        relevance_out *= self.m.in_tensor

        activator_relevances = self.f_linear(pw, nw, px, nx, relevance_in)
        inhibitor_relevances = self.f_linear(nw, pw, px, nx, relevance_in)

        relevance_out = self.alpha * activator_relevances - beta * inhibitor_relevances

        assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"
        return relevance_out
    
    




class LRP_transposeconv2d(RelProp):
    def __init__(self, conv2d, **kwargs):
        super(LRP_transposeconv2d, self).__init__()
        assert isinstance(conv2d,nn.Conv2d), "Please tie with a conv2d layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']


        self.m=conv2d 
        self.eps=10e-8
        self.in_tensor_shape=self.m.in_tensor.shape

        self.inv_m=torch.nn.ConvTranspose2d(in_channels=self.m.out_channels, out_channels=self.m.in_channels,
                                            kernel_size=self.m.kernel_size, stride=self.m.stride, padding=self.m.padding, 
                                            output_padding=self.output_pad(),groups=self.m.groups,bias=None) #bias=None


        # self.m.weight (c_out,c_in, k0,k1)
        # self.inv_m.weight (c_in, c_out, k0,k1)
        self.inv_m.weight=nn.Parameter(self.m.weight) # no need to do transpose(0,1)

    def output_pad(self):
        k=self.m.kernel_size
        s=self.m.stride
        p=self.m.padding
        expect_size=(self.in_tensor_shape[-2],self.in_tensor_shape[-1])
        ou_1=(expect_size[1]-k[1]+2*p[1])%s[1]
        ou_0=(expect_size[0]-k[0]+2*p[0])%s[0]
        return (ou_0,ou_1)

    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out

    def LRP_eps_forward(self,relevance_in):

        relevance_out = self.inv_m(safe_divide(relevance_in,self.m.out_tensor+self.eps*torch.mean(self.m.in_tensor)))
        #relevance_out =  self.inv_m(relevance_in/(self.m.out_tensor+self.eps*torch.mean(self.m.in_tensor)))
        relevance_out *= self.m.in_tensor
        assert not torch.isnan(relevance_out).any(), f"{self.inv_m} layer has nan output layer"
        return relevance_out

    def gradprop2(self, DY, weight):
        Z = self.m.forward(self.m.in_tensor)

        output_padding = self.m.in_tensor.size()[2] - (
                (Z.size()[2] - 1) * self.m.stride[0] - 2 * self.m.padding[0] + self.m.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.m.stride, padding=self.m.padding, output_padding=output_padding)

    def LRP_ab_forward(self,relevance_in):
        if self.m.in_tensor.shape[1] == 3:
            pw = torch.clamp(self.m.weight, min=0)
            nw = torch.clamp(self.m.weight, max=0)
            X = self.m.in_tensor
            L = self.m.in_tensor * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.m.in_tensor * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.m.weight, bias=None, stride=self.m.stride, padding=self.m.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.m.stride, padding=self.m.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.m.stride, padding=self.m.padding) + 1e-9

            S = relevance_in / Za
            relevance_out = X * self.gradprop2(S, self.m.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
        else:
            beta = self.alpha - 1
            pw = torch.clamp(self.m.weight, min=0)
            nw = torch.clamp(self.m.weight, max=0)
            px = torch.clamp(self.m.in_tensor, min=0)
            nx = torch.clamp(self.m.in_tensor, max=0)

            activator_relevances = self.f_conv2d(pw, nw, px, nx, relevance_in, stride=self.m.stride, padding=self.m.padding)     
            inhibitor_relevances = self.f_conv2d(nw, pw, px, nx, relevance_in, stride=self.m.stride, padding=self.m.padding)
            relevance_out = self.alpha * activator_relevances - beta * inhibitor_relevances
        

        return relevance_out

class LRP_BN2d( RelProp):
    def __init__(self, BN,  **kwargs):
        super(LRP_BN2d, self).__init__()
        assert isinstance(BN,nn.BatchNorm2d), "Please tie with a batchnorm2d function"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']


        self.m=BN
    

    def LRP_eps_forward(self,relevance_in):
        # scale=self.m.weight.reshape(1,-1,1,1)/torch.sqrt(self.m.running_var+self.m.eps).reshape(1,-1,1,1) # scale must be assigned at the forward pass
        # relevance_out =  scale*self.m.in_tensor/self.m.out_tensor*relevance_in
        relevance_out=relevance_in
        assert not torch.isnan(relevance_out).any(), f"invser_{self.m} layer has nan output."
        return relevance_out

    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out

    def LRP_ab_forward(self,relevance_in):

        # this is actually equal to silent pass relevance_in=relevance out TODO chekc if it is suitable
        X = self.m.in_tensor
        weight = self.m.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.m.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.m.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = relevance_in / Z
        Ca = S * weight
        relevance_out = self.m.in_tensor * (Ca)
        assert not torch.isnan(relevance_out).any(), f"{self.inv_m} layer has nan output layer"
        return relevance_out

    


class LRP_BN1d(RelProp):
    def __init__(self, BN, **kwargs):
        super(LRP_BN1d, self).__init__()
        assert isinstance(BN,nn.BatchNorm1d), "Please tie with a batchnorm1d layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']

        self.m=BN
    
    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out

    
    def LRP_eps_forward(self,relevance_in):
        scale=self.m.weight.reshape(1,-1)/torch.sqrt(self.m.running_var+self.m.eps).reshape(1,-1)
        relevance_out =  scale*self.m.in_tensor/self.m.out_tensor*relevance_in
        assert not torch.isnan(relevance_out).any(), f"invser_{self.m} layer has nan output relevance"
        return relevance_out

    def LRP_ab_forward(self,relevance_in):
        return relevance_in # TODO check how to do BN1D ab forward, as well as for BN2D




class LRP_avgpool2d( RelProp):
    def __init__(self, avgpool2d, **kwargs):
        super(LRP_avgpool2d,self).__init__()
        assert isinstance(avgpool2d,nn.AdaptiveAvgPool2d), "Please tie with an adaptive avgpool2d layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']

        self.m=avgpool2d
        self.const_weight=torch.ones(self.m.in_channels, 1,self.m.kernel_size[0],self.m.kernel_size[1])
        self.const_weight=self.const_weight/(prod(self.m.kernel_size))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.const_weight=self.const_weight.to(device)

    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out
      

    def LRP_eps_forward(self,relevance_in):
        # its reverse operation is F.conv_transpose2d with groups=in_channels
        # groups in transpose_conv2d https://medium.com/@reachraktim/convolutions-from-scratch-conv2d-transpose-convolution-group-convolution-depth-wise-a917f334c59c
        # no parameter need to be update so use function

        # F.interpolate equals to avgunpool?
        relevance_out=F.conv_transpose2d(relevance_in,weight=self.const_weight,bias=None, stride=self.m.stride,groups=self.m.in_channels)
        return relevance_out #TODO check

    def LRP_ab_forward(self,relevance_in):
        # refer to RelPropSImple class in https://github.com/shirgur/AGFVisualization/blob/64c01592b319c825b7d8f24b4b9d88f43fdcfa4b/modules/layers_rap.py#L51
        # theoreatically it should be same as the same way as LRP_eps_forward 
        Z = self.m.forward(self.m.in_tensor)
        S = safe_divide(relevance_in, Z)
        C = self.gradprop(Z, self.m.in_tensor, S)[0]

        if torch.is_tensor(self.m.in_tensor) == False: #TODO check when it will be called
            relevance_out = []
            relevance_out.append(self.m.in_tensor[0] * C)
            relevance_out.append(self.m.in_tensor[1] * C)
        else:
            relevance_out = self.m.in_tensor * (C)

        assert not torch.isnan(relevance_out).any(), f"invser_{self.m} layer has nan output."
        return relevance_out



class LRP_maxpool2d( RelProp):
    def __init__(self, maxpool2d, **kwargs):
        super(LRP_maxpool2d,self).__init__()
        assert isinstance(maxpool2d,nn.MaxPool2d), "Please tie with a maxpool layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']


        self.m=maxpool2d
    

    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out


    def LRP_eps_forward(self,relevance_in):
        # no parameter need to be update so use function
        relevance_out=F.max_unpool2d(relevance_in,self.m.indices,
                               self.m.kernel_size, self.m.stride,
                               self.m.padding, output_size=self.m.in_shape) #todo check padding attributes as well for LRP_avgpool2d
        return relevance_out #todo check


    def LRP_ab_forward(self,relevance_in): 
        # refer to RelPropSImple class in https://github.com/shirgur/AGFVisualization/blob/64c01592b319c825b7d8f24b4b9d88f43fdcfa4b/modules/layers_rap.py#L51
        Z = self.m.forward(self.m.in_tensor)
        S = safe_divide(relevance_in, Z)
        C = self.gradprop(Z, self.m.in_tensor, S)[0]

        if torch.is_tensor(self.m.in_tensor) == False: #TODO check when it will be called
            relevance_out = []
            relevance_out.append(self.m.in_tensor[0] * C)
            relevance_out.append(self.m.in_tensor[1] * C)
        else:
            relevance_out = self.m.in_tensor * (C)
  
        assert not torch.isnan(relevance_out).any(), f"invser_{self.m} layer has nan output."
        return relevance_out


