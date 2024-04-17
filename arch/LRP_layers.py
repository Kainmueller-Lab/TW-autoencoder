import torch
from torch import nn
from torch.nn import functional as F
from math import prod
import sys
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
        C = torch.autograd.grad(Z, X, S, retain_graph=False)
        return C

    def relprop(self, R, alpha):
        return R

    def f_linear_new(self,w, x, R):
        '''
        use this trick because all element in x is larger than 0
        '''
        if torch.all(x>=0):
            if self.me_eff:
                Z = F.linear(x,w).detach()
            else:
                Z = F.linear(x,w)
            S = safe_divide(R, Z)
            C = x* F.linear(S, w.t())
        else:
            raise NotImplementedError
        return C

    def f_linear(self,w1, w2, x1, x2, R):

        Z1 = F.linear(x1, w1)
        Z2 = F.linear(x2, w2)
        S1 = safe_divide(R, Z1)
        S2 = safe_divide(R, Z2)
        C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        C2 = x2 * self.gradprop(Z2, x2, S2)[0]

        return C1 + C2
    
    def f_conv2d_new(self, w, x, R):
        '''
        use this trick because all element in x is larger than 0
        '''
        if torch.all(x>=0):
            if self.me_eff:
                Z = F.conv2d(x, w, bias=None, stride=self.m.stride, padding=self.m.padding).detach()
            else:
                Z = F.conv2d(x, w, bias=None, stride=self.m.stride, padding=self.m.padding)
            S = safe_divide(R, Z)  
            C = x * F.conv_transpose2d(S, w, stride=self.m.stride, padding=self.m.padding, output_padding=self.cal_outpad())
        else:
            raise NotImplementedError
        return C

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
        self.me_eff=kwargs['memory_efficient']
        self.detach_bias=kwargs['detach_bias']
        self.normal_relu=kwargs['normal_relu']
        self.normal_deconv=kwargs['normal_deconv']
        self.remove_heaviside=kwargs['remove_heaviside']
        if self.xai!="LRP_epsilon":
            assert self.normal_deconv==False and self.normal_relu==False, "LRP_alphabeta/cLRP does not support normal_deconv and normal_relu, understruction"

        self.top_layer=kwargs.pop('top_layer', False)
        print(f"For LRP layer in Linear layer, 'detach bias' = {self.detach_bias}")

        if self.normal_deconv:
            self.inv_m=nn.Linear(in_features=linear.out_features, out_features=linear.in_features, bias=None) #bias=None
        else:
            self.m=linear       
            self.inv_m=nn.Linear(in_features=self.m.out_features, out_features=self.m.in_features, bias=None) #bias=None
            self.inv_m.weight=nn.Parameter(self.m.weight.t())

    def forward(self,relevance_in):
        if self.xai=="LRP_epsilon":
            relevance_out=self.LRP_eps_forward(relevance_in)
        elif self.xai=="LRP_alphabeta":
            relevance_out=self.LRP_ab_forward(relevance_in)
        elif self.xai=="cLRP_type1":
            self.alpha=1
            if self.top_layer:
                relevance_out=self.cLRP_type1_toplayer_forward(relevance_in)
            else:
                relevance_out=self.LRP_ab_forward(relevance_in)
        elif self.xai=="cLRP_type2":
            self.alpha=1
            if self.top_layer:
                relevance_out=self.cLRP_type2_toplayer_forward(relevance_in)
            else:
                relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out


    def LRP_eps_forward(self,relevance_in):
        if self.normal_deconv and self.normal_relu:
            relevance_out=self.inv_m(relevance_in)

        elif not self.normal_deconv and self.normal_relu:
            if self.top_layer:
                if self.remove_heaviside:
                    relevance_out=self.inv_m(relevance_in) # different from above as the definition of self.inv_m changes
                else:
                    # relevance_in[relevance_in!=0]=1.0
                    # relevance_out=self.inv_m(relevance_in) # different from above as the definition of self.inv_m changes
                    if self.m.bias==None or self.detach_bias==False:
                        out_tensor=self.m.forward(self.m.in_tensor)
                    else:
                        out_tensor = F.linear(self.m.in_tensor,self.m.weight,self.m.bias.detach()) # change on 13/07/2023
                    relevance_out = self.inv_m(safe_divide(relevance_in,out_tensor+self.eps))
            else:
                relevance_out=self.inv_m(relevance_in) # different from above as the definition of self.inv_m changes
        
        else:
            if self.me_eff:
                self.m.in_tensor=self.m.in_tensor.detach()
                out_tensor=self.m.forward(self.m.in_tensor).detach()
            else:
                if self.m.bias==None or self.detach_bias==False:
                    out_tensor=self.m.forward(self.m.in_tensor)
                else:
                    out_tensor = F.linear(self.m.in_tensor,self.m.weight,self.m.bias.detach()) # change on 13/07/2023
            relevance_out = self.inv_m(safe_divide(relevance_in,out_tensor+self.eps))
            relevance_out *= self.m.in_tensor
    

        assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"
        return relevance_out

    # 17.02.2023
    def LRP_ab_forward(self,relevance_in):
        if self.me_eff:
            self.m.in_tensor=self.m.in_tensor.detach()
        beta=self.alpha-1
        pw = torch.clamp(self.m.weight, min=0)
        nw = torch.clamp(self.m.weight, max=0)
#         px = torch.clamp(self.m.in_tensor, min=0)
#         nx = torch.clamp(self.m.in_tensor, max=0)

#         activator_relevances = self.f_linear(pw, nw, px, nx, relevance_in)
#         inhibitor_relevances = self.f_linear(nw, pw, px, nx, relevance_in)


        activator_relevances = self.f_linear_new(pw, self.m.in_tensor, relevance_in)
        inhibitor_relevances = self.f_linear_new(nw, self.m.in_tensor, relevance_in)
        # print("--------------------verify--------------")
        # print(torch.all(activator_relevances1==activator_relevances2))


        # relevance_out1 = self.alpha * activator_relevances1 - beta * inhibitor_relevances1
        if beta==0:
            relevance_out = self.alpha * activator_relevances
        else:
            relevance_out = self.alpha * activator_relevances - beta * inhibitor_relevances

        assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"

        return relevance_out

    def cLRP_type1_toplayer_forward(self, relevance_in):
        '''
        initialize the top layer relevance distribution as 
        R_tgt=1 * class_score(tgt)
        R_rest= -1/(N-1) * class_score(tgt)
        so that sum_j R_j=0 
        '''
        num_classes=relevance_in.shape[1]
        if self.m.bias==None or self.detach_bias==False:
            Y = self.m.forward(self.m.in_tensor)
        else:
            Y = F.linear(self.m.in_tensor,self.m.weight,self.m.bias.detach()) # change on 13/07/2023
        R_rest=torch.abs(Y- relevance_in)
        R_tgt=torch.abs(relevance_in)
        tmp_relevance=-1*safe_divide(R_rest,R_rest.sum(dim=1, keepdim=True).detach())*R_tgt.sum(dim=1, keepdim=True).detach()+R_tgt

        if self.me_eff:
            self.m.in_tensor=self.m.in_tensor.detach()
        pw = torch.clamp(self.m.weight, min=0)
        activator_relevances = self.f_linear_new(pw, self.m.in_tensor, tmp_relevance)
        relevance_out =  activator_relevances
        assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"

        return relevance_out

    def cLRP_type2_toplayer_forward(self, relevance_in):
        '''
        initialize the top layer relevance distribution as 
        R_tgt= class_score[:, tgt]
        R_rest= 0
        but do two times forward with each time only pick positive contribution connection or only pick negative contribution connection
        '''
        if self.me_eff:
            self.m.in_tensor=self.m.in_tensor.detach()
        pw = torch.clamp(self.m.weight, min=0)
        nw = torch.clamp(self.m.weight, max=0)
        activator_relevances = self.f_linear_new(pw, self.m.in_tensor, torch.abs(relevance_in))
        inhibitor_relevances = self.f_linear_new(nw, self.m.in_tensor, torch.abs(relevance_in))
        relevance_out = activator_relevances- inhibitor_relevances
        assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"

        return relevance_out

    # def LRP_ab_forward(self,relevance_in):
    #     beta=self.alpha-1
    #     pw = torch.clamp(self.m.weight, min=0)
    #     nw = torch.clamp(self.m.weight, max=0)
    #     px = torch.clamp(self.m.in_tensor, min=0)
    #     nx = torch.clamp(self.m.in_tensor, max=0)
     
    #     relevance_out = self.inv_m(relevance_in/(self.m.out_tensor+self.eps))
    #     relevance_out *= self.m.in_tensor

    #     activator_relevances = self.f_linear(pw, nw, px, nx, relevance_in)
    #     inhibitor_relevances = self.f_linear(nw, pw, px, nx, relevance_in)

    #     relevance_out = self.alpha * activator_relevances - beta * inhibitor_relevances

    #     assert not torch.any(relevance_out.isnan()), f"{self.inv_m} layer has nan output relevance"
    #     return relevance_out


    
    




class LRP_transposeconv2d(RelProp):
    def __init__(self, conv2d, **kwargs):
        super(LRP_transposeconv2d, self).__init__()
        assert isinstance(conv2d,nn.Conv2d), "Please tie with a conv2d layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']
        self.me_eff=kwargs['memory_efficient']
        self.detach_bias=kwargs['detach_bias']

        self.normal_relu=kwargs['normal_relu']
        self.normal_deconv=kwargs['normal_deconv']
        if self.xai!="LRP_epsilon":
            assert self.normal_deconv==False and self.normal_relu==False, "LRP_alphabeta/cLRP does not support normal_deconv and normal_relu, understruction"

       


        self.m=conv2d 
        self.in_tensor_shape=self.m.in_tensor.shape

        if self.normal_deconv:
            # if it is tied with the first layer in the encoder, we set the output channels=21
            if self.m.in_tensor.shape[1] == 3:
                self.inv_m=torch.nn.ConvTranspose2d(in_channels=self.m.out_channels, out_channels=21,
                                                kernel_size=self.m.kernel_size, stride=self.m.stride, padding=self.m.padding, 
                                                output_padding=self.output_pad(),groups=self.m.groups,bias=True) #bias=None
                print(f"First Free tranposeconv2d layer, out_channels=21, bias=True")
            else:
                self.inv_m=torch.nn.ConvTranspose2d(in_channels=self.m.out_channels, out_channels=self.m.in_channels,
                                                    kernel_size=self.m.kernel_size, stride=self.m.stride, padding=self.m.padding, 
                                                    output_padding=self.output_pad(),groups=self.m.groups,bias=True) #bias=None
                print(f"Free tranposeconv2d layer, bias=True")
        
        else:
            self.inv_m=torch.nn.ConvTranspose2d(in_channels=self.m.out_channels, out_channels=self.m.in_channels,
                                                kernel_size=self.m.kernel_size, stride=self.m.stride, padding=self.m.padding, 
                                                output_padding=self.output_pad(),groups=self.m.groups,bias=None) #bias=None
            # self.m.weight (c_out,c_in, k0,k1)
            # self.inv_m.weight (c_in, c_out, k0,k1)
            self.inv_m.weight=nn.Parameter(self.m.weight) # no need to do transpose(0,1)
            print(f"Tied tranposeconv2d layer, 'detach bias' = {self.detach_bias}")

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
        elif self.xai in ["cLRP_type1","cLRP_type2"]:
            self.alpha=1
            relevance_out=self.LRP_ab_forward(relevance_in)
        return relevance_out

    def LRP_eps_forward(self,relevance_in):
        if self.normal_deconv and self.normal_relu:
            relevance_out=self.inv_m(relevance_in)

        elif not self.normal_deconv and self.normal_relu:
            relevance_out=self.inv_m(relevance_in) # different from above as the definition of self.inv_m changes
        
        else:
            if self.me_eff:
                self.m.in_tensor=self.m.in_tensor.detach()
                out_tensor=self.m.forward(self.m.in_tensor).detach()
            else:
                if self.m.bias==None or self.detach_bias==False:
                    out_tensor=self.m.forward(self.m.in_tensor) # change on 16/07/2023 (find bug)
                else:
                    out_tensor=F.conv2d(self.m.in_tensor,weight=self.m.weight, bias=self.m.bias.detach(), stride=self.m.stride, padding=self.m.padding) # change on 13/07/2023
            relevance_out = self.inv_m(safe_divide(relevance_in,out_tensor+self.eps))
            relevance_out *= self.m.in_tensor
            assert not torch.isnan(relevance_out).any(), f"{self.inv_m} layer has nan output layer"
        return relevance_out
    
    def cal_outpad(self):
        Z = self.m.forward(self.m.in_tensor).detach()

        output_padding = self.m.in_tensor.size()[2] - (
                (Z.size()[2] - 1) * self.m.stride[0] - 2 * self.m.padding[0] + self.m.kernel_size[0])
        return output_padding
    
    def gradprop2(self, DY, weight):
        output_padding=self.cal_outpad()
        return F.conv_transpose2d(DY, weight, stride=self.m.stride, padding=self.m.padding, output_padding=output_padding)

    def LRP_ab_forward(self,relevance_in):
        if self.m.in_tensor.shape[1] == 3:
            if self.me_eff:
                X = self.m.in_tensor.detach()
            else:
                X = self.m.in_tensor
                
            pw = torch.clamp(self.m.weight, min=0)
            nw = torch.clamp(self.m.weight, max=0)
            L = self.m.in_tensor * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.m.in_tensor * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.m.weight, bias=None, stride=self.m.stride, padding=self.m.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.m.stride, padding=self.m.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.m.stride, padding=self.m.padding) + 1e-9
            if self.me_eff:
                S = relevance_in / Za.detach()
            else:
                S = relevance_in / Za
            relevance_out = X * self.gradprop2(S, self.m.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            del pw, nw ,X, L, H, Za, S
        else:
            if self.me_eff:
                self.m.in_tensor = self.m.in_tensor.detach()
            beta = self.alpha - 1
            pw = torch.clamp(self.m.weight, min=0)
            nw = torch.clamp(self.m.weight, max=0)
#             px = torch.clamp(self.m.in_tensor, min=0)
#             nx = torch.clamp(self.m.in_tensor, max=0)
            
#             activator_relevances = self.f_conv2d(pw, nw, px, nx, relevance_in, stride=self.m.stride, padding=self.m.padding)     
#             inhibitor_relevances = self.f_conv2d(nw, pw, px, nx, relevance_in, stride=self.m.stride, padding=self.m.padding)


            activator_relevances = self.f_conv2d_new(pw, self.m.in_tensor, relevance_in)     
            inhibitor_relevances = self.f_conv2d_new(nw, self.m.in_tensor, relevance_in)

            # # print("--------------------verify--------------")
            # # print(torch.all(activator_relevances1==activator_relevances2),"actiivator_relevances1")

            if beta==0:
                relevance_out = self.alpha * activator_relevances
            else:
                relevance_out = self.alpha * activator_relevances - beta * inhibitor_relevances
            # print("--------------------verify--------------")
            # print(torch.all(relevance_out1==relevance_out2),"relevance_out")
            # if torch.all(relevance_out1==relevance_out2)==False:
            #     print((relevance_out1-relevance_out2).max(), "is the maxmimum difference")
            
            del  nw, pw,activator_relevances,inhibitor_relevances
        return relevance_out

class LRP_BN2d( RelProp):
    def __init__(self, BN,  **kwargs):
        super(LRP_BN2d, self).__init__()
        assert isinstance(BN,nn.BatchNorm2d), "Please tie with a batchnorm2d function"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']
        self.me_eff=kwargs['memory_efficient']

        self.normal_relu=kwargs['normal_relu']
        self.normal_deconv=kwargs['normal_deconv']
        if self.xai!="LRP_epsilon":
            assert self.normal_deconv==False and self.normal_relu==False, "LRP_alphabeta/cLRP does not support normal_deconv and normal_relu, understruction"

        self.m=BN

        if self.normal_deconv:
            self.batch_norm=nn.BatchNorm2d(self.m.num_features)

    

    def LRP_eps_forward(self,relevance_in):
        if self.normal_deconv and self.normal_relu:
            relevance_out=self.batch_norm(relevance_in)
        
        elif not self.normal_deconv and self.normal_relu:
            if self.m.training:
                scale=safe_divide(self.m.weight.reshape(1,-1,1,1),torch.sqrt(torch.var(self.m.in_tensor,dim=(0,-2,-1),keepdim=True,unbiased=False) 
                                +self.m.eps).reshape(1,-1,1,1)).detach()
            else:
                scale=safe_divide(self.m.weight.reshape(1,-1,1,1),torch.sqrt(self.m.running_var+self.m.eps).reshape(1,-1,1,1)).detach()
            relevance_out=scale*relevance_in

        else:
            # self.test_BN_property()
            if self.m.training: #change on 09/07/2023
                # torch.var need to set the unbiased=False
                scale=safe_divide(self.m.weight.reshape(1,-1,1,1),torch.sqrt(torch.var(self.m.in_tensor,dim=(0,-2,-1),keepdim=True,unbiased=False) 
                                +self.m.eps).reshape(1,-1,1,1)).detach()
                
            else:
                scale=safe_divide(self.m.weight.reshape(1,-1,1,1),torch.sqrt(self.m.running_var+self.m.eps).reshape(1,-1,1,1)).detach() # scale must be assigned at the forward pass
                
            
            relevance_out =  scale* safe_divide(self.m.in_tensor,self.m.out_tensor).detach()*relevance_in # change on 09/07/2023
        
            
            assert not torch.isnan(relevance_out).any(), f"invser_{self.m} layer has nan output {relevance_out.max()} {relevance_out.min()}{in_out_ratio.max()} {in_out_ratio.min()} {scale.max()} {scale.min()}."
        return relevance_out

        # if self.m.training: #change on 09/07/2023
        #     # torch.var need to set the unbiased=False
        #     scale=safe_divide(self.m.weight.reshape(1,-1,1,1),torch.sqrt(torch.var(self.m.in_tensor,dim=(0,-2,-1),keepdim=True,unbiased=False) 
        #                     +self.m.eps).reshape(1,-1,1,1)).detach()
            
        # else:
        #     scale=safe_divide(self.m.weight.reshape(1,-1,1,1),torch.sqrt(self.m.running_var+self.m.eps).reshape(1,-1,1,1)).detach() # scale must be assigned at the forward pass
            
        # assert not torch.isnan(scale).any(), f"invser_{self.m} layer has scale nan output. {scale.max()} {scale.min()}"
        # in_out_ratio=safe_divide(self.m.in_tensor,self.m.out_tensor)
        # assert not torch.isnan(in_out_ratio).any(), f"invser_{self.m} layer has nan output {in_out_ratio.max()} {in_out_ratio.min()}."
        # relevance_out =  scale* safe_divide(self.m.in_tensor,self.m.out_tensor).detach()*relevance_in # change on 09/07/2023
       
        
        # assert not torch.isnan(relevance_out).any(), f"invser_{self.m} layer has nan output {relevance_out.max()} {relevance_out.min()}{in_out_ratio.max()} {in_out_ratio.min()} {scale.max()} {scale.min()}."
        # return relevance_out
        

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
        self.me_eff=kwargs['memory_efficient']


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
        self.me_eff=kwargs['memory_efficient']

        self.normal_relu=kwargs['normal_relu']
        self.normal_deconv=kwargs['normal_deconv']


        self.m=avgpool2d
        self.const_weight=torch.ones(self.m.in_channels, 1,self.m.kernel_size[0],self.m.kernel_size[1])
        self.const_weight=self.const_weight/(prod(self.m.kernel_size))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.const_weight=self.const_weight.to(device)

    def forward(self,relevance_in):
        # if self.xai=="LRP_epsilon":
        #     relevance_out=self.LRP_eps_forward(relevance_in)
        # elif self.xai=="LRP_alphabeta":
            # relevance_out=self.LRP_ab_forward(relevance_in)
        relevance_out=self.LRP_eps_forward(relevance_in)
        return relevance_out
      

    def LRP_eps_forward(self,relevance_in):
        if self.normal_deconv or self.normal_relu:
            relevance_out=F.conv_transpose2d(relevance_in,weight=self.const_weight,bias=None, stride=self.m.stride,groups=self.m.in_channels)
        else:
            # its reverse operation is F.conv_transpose2d with groups=in_channels
            # groups in transpose_conv2d https://medium.com/@reachraktim/convolutions-from-scratch-conv2d-transpose-convolution-group-convolution-depth-wise-a917f334c59c
            # no parameter need to be update so use function
            # F.interpolate equals to avgunpool?
            if self.me_eff:
                self.m.in_tensor=self.m.in_tensor.detach()
                out_tensor=self.m.forward(self.m.in_tensor).detach()
            else:
                out_tensor=self.m.forward(self.m.in_tensor)
            relevance_out=self.m.in_tensor*F.conv_transpose2d(safe_divide(relevance_in,out_tensor),weight=self.const_weight,bias=None, stride=self.m.stride,groups=self.m.in_channels)
        return relevance_out # debug on 17.02.2023

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
        self.me_eff=kwargs['memory_efficient']
        self.normal_unpool=kwargs['normal_unpool']
        if self.normal_unpool:
            print("Using nearest interpolation for unpooling")
        else:
            print(f"Tied max-unpooling layer.")

        # no change need for ablation test setting which set normal_relu or normal_deconv==True



        self.m=maxpool2d
    

    def forward(self,relevance_in):
        # if self.xai=="LRP_epsilon":
        #     relevance_out=self.LRP_eps_forward(relevance_in)
        # elif self.xai=="LRP_alphabeta":
            #no matter alpha bata rule or epsilon rule, inv_maxpool should be same
        relevance_out=self.LRP_eps_forward(relevance_in)
       
            
        return relevance_out


    def LRP_eps_forward(self,relevance_in):
        # no parameter need to be update so use function
        if self.normal_unpool:
            relevance_out=F.interpolate(relevance_in,scale_factor=self.m.stride,mode="nearest")
        else:
            relevance_out=F.max_unpool2d(relevance_in,self.m.indices,
                                self.m.kernel_size, self.m.stride,
                                self.m.padding, output_size=self.m.in_shape) #todo check padding attributes as well for LRP_avgpool2d
        
        return relevance_out # prove to be right implementation

    ##################################################################################
    # # This part is just used to verify the gradients don't flow back to encoder
    # # Please comment the following code TODO

    # def cal_outpad(self):

    #     output_padding = self.m.in_shape[2] - (
    #             (self.m.out_shape[2] - 1) * self.m.stride - 2 * self.m.padding + self.m.kernel_size)
    #     return output_padding


    # def LRP_eps_forward(self,relevance_in):
    #     # no parameter need to be update so use function
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     const_weight=torch.ones(self.m.in_shape[1], 1,self.m.kernel_size,self.m.kernel_size).to(device)
    #     out_tensor=self.m.forward(self.m.in_tensor)
    #     relevance_out=self.m.in_tensor*F.conv_transpose2d(safe_divide(relevance_in,out_tensor),
    #         weight=const_weight,bias=None, padding=self.m.padding, stride=self.m.stride,
    #             groups=self.m.in_shape[1],output_padding=self.cal_outpad())
       
    #     return relevance_out # prove to be right implementation
    ##################################################################################

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


# LRP_bottle_conv is used to tie with the linear layer
class LRP_bottle_conv(RelProp):
    def __init__(self, linear,avgpool, **kwargs):
        super(LRP_bottle_conv, self).__init__()
        assert isinstance(linear,nn.Linear), "Please tie with a linear layer and an avgpool layer"
        self.xai=kwargs['xai']
        self.eps=kwargs['epsilon']
        self.alpha=kwargs['alpha']
        self.me_eff=kwargs['memory_efficient']
        self.detach_bias=kwargs['detach_bias']
        self.normal_relu=kwargs['normal_relu']
        self.normal_deconv=kwargs['normal_deconv']
        self.remove_heaviside=kwargs['remove_heaviside']
        self.add_bottle_conv=kwargs['add_bottle_conv']
        if self.xai!="LRP_epsilon":
            assert self.normal_deconv==False and self.normal_relu==False, "LRP_alphabeta/cLRP does not support normal_deconv and normal_relu, understruction"

        self.top_layer=kwargs.pop('top_layer', False)
        print(f"For LRP layer in Linear layer, 'detach bias' = {self.detach_bias}")
        self.m=linear
        self.window_size=prod(avgpool.kernel_size)

    def forward(self,relevance_in,class_idx):
        if self.normal_deconv:
            if self.add_bottle_conv:
                weight=self.m.weight.sum(dim=0)
                weight=(weight/self.window_size).reshape(-1,1,1,1)
                relevance_out= F.conv2d(relevance_in, weight=weight, bias=None, groups=relevance_in.shape[1])
            else:
                relevance_out= relevance_in
        else:
            # the shape of linear.weight [out_features, in_features] =[21, num_features]
            weight=self.m.weight[class_idx,:]
            weight=(weight/self.window_size).reshape(-1,1,1,1)
            relevance_out= F.conv2d(relevance_in, weight=weight, bias=None, groups=relevance_in.shape[1])
        return relevance_out

