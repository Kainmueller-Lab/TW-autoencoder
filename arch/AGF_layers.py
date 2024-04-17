import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod
import sys

__all__ = ['forward_hook', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'safe_divide',
           'Add', 'Clone', 'minmax_dims']


# Misc
def get_factorization(X, grad, phi):
    # Phi partition
    phi_pos = phi.clamp(min=0)
    phi_neg = phi.clamp(max=0)

    # Normalize inputs
    grad = safe_divide(grad, minmax_dims(grad, 'max'))
    X = safe_divide(X, minmax_dims(X, 'max'))

    # Compute F_dx
    # Heaviside function
    ref = torch.sigmoid(10 * grad)

    # Compute representatives - R
    R_pos = safe_divide((ref * phi_pos.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_pos.ne(0).type(phi_pos.type()).sum(dim=[1, 2, 3], keepdim=True)))
    R_neg = safe_divide((ref * phi_neg.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_neg.ne(0).type(phi_neg.type()).sum(dim=[1, 2, 3], keepdim=True)))
    R = torch.cat((R_neg.squeeze(3), R_pos.squeeze(3)), dim=-1)

    # Compute weights - W
    H = ref.reshape(X.shape[0], X.shape[1], -1)
    W = ((R.permute(0, 2, 1) @ R + torch.eye(2)[None, :, :].cuda() * 1) * 1).inverse() @ (R.permute(0, 2, 1) @ H)
    W = F.relu(W.reshape(W.shape[0], W.shape[1], X.shape[2], X.shape[3]))
    F_dx = -(W[:, 0:1] - W[:, 1:2])
    F_dx = F_dx.expand_as(X)

    # Compute F_x
    # Heaviside function
    ref = torch.sigmoid(10 * X)

    # Compute representatives - R
    R_pos = safe_divide((ref * phi_pos.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_pos.ne(0).type(phi_pos.type()).sum(dim=[1, 2, 3], keepdim=True)))
    R_neg = safe_divide((ref * phi_neg.ne(0).type(ref.type())).sum(dim=[2, 3], keepdim=True), (
        phi_neg.ne(0).type(phi_neg.type()).sum(dim=[1, 2, 3], keepdim=True)))

    # Compute weights - W
    R = torch.cat((R_neg.squeeze(3), R_pos.squeeze(3)), dim=-1)
    H = ref.reshape(X.shape[0], X.shape[1], -1)
    W = ((R.permute(0, 2, 1) @ R + torch.eye(2)[None, :, :].cuda() * 1) * 1).inverse() @ (R.permute(0, 2, 1) @ H)
    W = F.relu(W.reshape(W.shape[0], W.shape[1], X.shape[2], X.shape[3]))
    F_x = -(W[:, 0:1] - W[:, 1:2])
    F_x = F_x.expand_as(X)

    return F_x, F_dx


def reg_scale(a, b):
    dim_range = list(range(1, a.dim()))
    return a * safe_divide(b.sum(dim=dim_range, keepdim=True), a.sum(dim=dim_range, keepdim=True))


def minmax_dims(x, minmax):
    y = x.clone()
    dims = x.dim()
    for i in range(1, dims):
        y = getattr(y, minmax)(dim=i, keepdim=True)[0]
    return y


def safe_divide(a, b):
    '''
    a=torch.tensor([1,0,0,1])
    b=torch.tensor([0,0,1,1])
    
    safe_divide(a,b): torch.tensor([0,0,0,1])
    '''
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())





def delta_shift(C, R):
    dim_range = list(range(1, R.dim()))
    nonzero = C.ne(0).type(C.type())
    result = C + R - (R.sum(dim=dim_range, keepdim=True)) / (nonzero.sum(dim=dim_range, keepdim=True)) * nonzero

    return result


# --------------------------------------
# Basic class 
# --------------------------------------
class AGFProp(nn.Module):
    def __init__(self):
        super(AGFProp, self).__init__()
        

    def AGF(self, m, cam=None, grad_outputs=None, **kwargs):
        # Gradient
        Y = m.forward(m.X)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, m.X, S,retain_graph=True)

        return cam, grad_out


class AGFPropSimple(AGFProp):


    def AGF(self, m, cam=None, grad_outputs=None, **kwargs):
        def backward(cam):
            X=m.in_tensor
            Z = m.forward(X) * cam.ne(0).type(cam.type())
            S = safe_divide(cam, Z)

            if torch.is_tensor(X) == False:
                # TODO: check
                result = []
                grad = torch.autograd.grad(Z, X, S,retain_graph=True)
                result.append(X[0] * grad[0])
                result.append(X[1] * grad[1])
            else:
                grad = torch.autograd.grad(Z, X, S,retain_graph=True)
                result = X * grad[0]
            return result

        if torch.is_tensor(cam) == False:
            idx = len(cam)
            tmp_cam = cam
            result = []
            for i in range(idx):
                cam_tmp = backward(tmp_cam[i])
                result.append(cam_tmp)
        else:
            result = backward(cam)

        # Gradient
        X=m.in_tensor
        Y = m.forward(X)
        S = grad_outputs
        #grad_out = torch.autograd.grad(Y, m.X, S)
        grad_out = torch.autograd.grad(Y, X, S,retain_graph=True)[0]

        return result, grad_out

# --------------------------------------
# Layers for AGF
# --------------------------------------

# TODO check how to deal with this layer, especially in resnet
class AGF_add(AGFPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

# TODO check how to deal with this layer, especially in resnet
class AGF_clone(AGFProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)

        S = []

        for z, c in zip(Z, cam):
            S.append(safe_divide(c, z))

        C = torch.autograd.grad(Z, self.X, S,retain_graph=True)

        R = self.X * C[0]

        # Gradient
        Y = self.forward(self.X, self.num)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S,retain_graph=True)

        return R, grad_out

# TODO check how to deal with this layer, especially in resnet
class AGF_cat(AGFProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def AGF(self, cam=None, grad_outputs=None, **kwargs):
        raise NotImplemented


class AGF_relu(AGFPropSimple):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.ReLU), "Please tie with a relu layer"
        self.m=m
    
    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
        return self.AGF(self.m, cam, grad_outputs, **kwargs)


class AGF_maxpool2d( AGFPropSimple):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.MaxPool2d), "Please tie with a maxpool2d layer"
        self.m=m
        self.me_eff=kwargs['memory_efficient']

    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]

        # TODO check why there is case that cam1-cam2>1e-4, while most of value equal
        # cam1,grad1=self.AGF(self.m, cam, grad_outputs, **kwargs)
        # cam2,grad2=self.forward_agf_maxpool(cam, grad_outputs, **kwargs)
        # print("R_out yours",cam1[0,0,:10,:10])
        # print("R_out mine",cam2[0,0,:10,:10])
        # print((cam1-cam2)[0,0,:10,:10])
        # print(torch.all(cam1==cam2))
        # print((cam1-cam2).max())
        # print(f"{((cam1-cam2)[0,0,10,10].abs().max())} is the cam maxmimum difference")
        # if (cam1-cam2).abs().max()>1e-4:
        #     print((cam1-cam2).max())
        #     sys.exit()
        # print(f"{((grad1-grad2).max())} is the gradmaxmimum difference")
        # return self.forward_agf_maxpool(cam, grad_outputs, **kwargs)
        return self.forward_agf_maxpool(cam, grad_outputs, **kwargs)

    def forward_agf_maxpool(self, cam=None, grad_outputs=None, **kwargs):
        def backward(cam):
            if torch.is_tensor(self.m.in_tensor) == False:
                raise NotImplementedError
            else: 
                # out_tensor=self.m.forward(self.m.in_tensor)
                # result=self.m.in_tensor*F.max_unpool2d(safe_divide(cam,out_tensor),self.m.indices,
                #                 self.m.kernel_size, self.m.stride,
                #                 self.m.padding, output_size=self.m.in_shape) 
                result=F.max_unpool2d(cam,self.m.indices,
                                self.m.kernel_size, self.m.stride,
                                self.m.padding, output_size=self.m.in_shape) 
            return result

        if torch.is_tensor(cam) == False:
            idx = len(cam)
            tmp_cam = cam
            result = []
            for i in range(idx):
                cam_tmp = backward(tmp_cam[i])
                result.append(cam_tmp)
        else:
            result=backward(cam)

        # Gradient
        grad_out=F.max_unpool2d(grad_outputs,self.m.indices,
                               self.m.kernel_size, self.m.stride,
                               self.m.padding, output_size=self.m.in_shape) 
        
        return result, grad_out


class AGF_dropout(AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Dropout), "Please tie with a dropout layer"
        self.m=m
        self.me_eff=kwargs['memory_efficient']


    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
        return self.AGF(self.m, cam, grad_outputs, **kwargs)



class AGF_BN2d(AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.BatchNorm2d), "Please tie with a batchnorm2d layer"
        self.m=m
        self.me_eff=kwargs['memory_efficient']


    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
        return self.AGF(self.m, cam, grad_outputs, **kwargs)

    


class AGF_avgpool2d(AGFPropSimple):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.AdaptiveAvgPool2d), "Please tie with an adaptive avgpool2d layer"
        self.m=m
        self.me_eff=kwargs['memory_efficient']


        self.const_weight=torch.ones(self.m.in_channels, 1,self.m.kernel_size[0],self.m.kernel_size[1])
        self.const_weight=self.const_weight/(prod(self.m.kernel_size))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.const_weight=self.const_weight.to(device)

    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]

        
        # cam1,grad1=self.AGF(self.m, cam, grad_outputs, **kwargs)
        # cam2,grad2=self.forward_agf_avgpool(cam, grad_outputs, **kwargs)
        # print("R_out yours",cam1[0,0,:,:])
        # print("R_out mine",cam2[0,0,:,:])
        # print((cam1-cam2)[0,0,:,:])
        # print(torch.all(cam1==cam2))
        # print(torch.all(grad1==grad2))
        # print(f"{((cam1-cam2).max())} is the cam maxmimum difference")
        # print(f"{((grad1-grad2).max())} is the cam maxmimum difference")
        # if (cam1-cam2).abs().max()>1e-4:
        #     sys.exit()
        # print(f"{((grad1-grad2).max())} is the gradmaxmimum difference")
        return self.forward_agf_avgpool(cam, grad_outputs, **kwargs)

    def forward_agf_avgpool(self, cam=None, grad_outputs=None, **kwargs):
        def backward(cam):
            if torch.is_tensor(self.m.in_tensor) == False:
                raise NotImplementedError
            else: 
                if self.me_eff:
                    self.m.in_tensor=self.m.in_tensor.detach()
                    out_tensor=self.m.forward(self.m.in_tensor).detach()
                else:
                    out_tensor=self.m.forward(self.m.in_tensor)
                result=self.m.in_tensor*F.conv_transpose2d(safe_divide(cam,out_tensor),
                        weight=self.const_weight,bias=None, stride=self.m.stride,groups=self.m.in_channels)
               
            return result

        if torch.is_tensor(cam) == False:
            idx = len(cam)
            tmp_cam = cam
            result = []
            for i in range(idx):
                cam_tmp = backward(tmp_cam[i])
                result.append(cam_tmp)
        else:
            result=backward(cam)

        # Gradient
        grad_out=F.conv_transpose2d(grad_outputs,
                        weight=self.const_weight,bias=None, stride=self.m.stride,groups=self.m.in_channels)
        
        return result, grad_out
   


# class AvgPool2d(nn.AvgPool2d, AGFPropSimple):
#     pass


class AGF_linear( AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Linear), "Please tie with a linear layer"
        self.m=m
        self.me_eff=kwargs['memory_efficient']

        
        # should be deleted later
        self.inv_m=nn.Linear(in_features=self.m.out_features, out_features=self.m.in_features, bias=None) #bias=None
        self.inv_m.weight=nn.Parameter(self.m.weight.t())

    def forward(self, cam=None, grad_outputs=None, **kwargs):
        #TODO chekc if it will have conflict for resnet 16.02.2023
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]

        
        if grad_outputs is None: # for the first backwards layer
            # Salient
            with torch.enable_grad():
                Y = self.m.forward(self.m.X)
            
                if 'K' in kwargs.keys():
                    target_class = Y.data.topk(kwargs['K'], dim=1)[1]
                else:
                    target_class = Y.data.topk(1, dim=1)[1] # N x 1
            
                if 'index' in kwargs.keys():
                    target_class = target_class[:, kwargs['index']:kwargs['index'] + 1]
                if 'class_id' in kwargs.keys():
                    if type(kwargs['class_id']) is list:
                        assert len(kwargs['class_id']) == len(target_class)
                        for i in range(len(kwargs['class_id'])):
                            target_class[i, 0] = kwargs['class_id'][i]
                    else:
                        raise Exception('Must be a list')


                # Initial propagation
                tgt = torch.zeros_like(Y) # N x C
                tgt = tgt.scatter(1, target_class, 1)
                yt = Y.gather(1, target_class) # N x 1

                sigma = (Y - yt).abs().max(dim=1, keepdim=True)[0]
    #             Y = F.softmax(yt * torch.exp(-0.5 * ((Y - yt) / sigma) ** 2), dim=1)
                Y1 = F.softmax(yt * torch.exp(-0.5 * ((Y - yt) / sigma) ** 2), dim=1)
                # Gradients stream
                # grad_out = torch.autograd.grad(Y, self.m.X, tgt,retain_graph=True)[0]
                # grad_out = torch.autograd.grad(Y, self.m.X, tgt) #TODO check
                if self.me_eff:
                    grad_out = F.linear(torch.autograd.grad(Y1, Y, tgt)[0].detach(), self.m.weight.t())
                else:
                    grad_out = torch.autograd.grad(Y1, self.m.X, tgt, create_graph=True)[0]
                #result = self.m.X * grad_out[0] #TODO check
                result = self.m.X * grad_out
        else:
            if self.me_eff:
                self.m.X=self.m.X.detach()
            # Compute - C
            xabs = self.m.X if torch.all(self.m.X>=0) else self.m.X.abs()
            wabs = self.m.weight.abs()
            Zabs = F.linear(xabs, wabs) * cam.ne(0).type(cam.type())
            
            if self.me_eff:
                Zabs=Zabs.detach()
            S = safe_divide(cam, Zabs)
            # grad = torch.autograd.grad(Zabs, xabs, S,retain_graph=True)
            grad=F.linear(S , wabs.t())
            # C = xabs * grad[0]
            C = xabs * grad

            # Compute - M
            # Y = self.m.forward(self.m.X)
            # S = grad_outputs[0]
            S = grad_outputs #TODO check

          
            # grad = torch.autograd.grad(Y, self.m.X, S,retain_graph=True)
            grad= F.linear(S,self.m.weight.t())
            if self.m.reshape_gfn is not None:
                x = self.m.reshape_gfn(self.m.X)
                # g = self.m.reshape_gfn(grad[0])
                g = self.m.reshape_gfn(grad)
                M = x * g
                M = M.mean(dim=1, keepdim=True).expand_as(x)
                M = M.reshape_as(self.m.X)
                M = F.relu(M) * C.ne(0).type(C.type())
                M = safe_divide(M, minmax_dims(M, 'max'))

                # Delta shift
                result = delta_shift(C, M)
            else:
                result = C

            # Gradients stream
            # Y = self.m.forward(self.m.X)
            S = grad_outputs
            # grad_out = torch.autograd.grad(Y, self.m.X, S)
            # grad_out = torch.autograd.grad(Y, self.m.X, S,retain_graph=True)[0]
            grad_out=F.linear(S, self.m.weight.t())

      
        return result, grad_out


class AGF_transposeconv2d( AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Conv2d), "Please tie with a conv2d layer"
        self.m=m
        self.me_eff=kwargs['memory_efficient']


        # should be deleted later
        self.inv_m=torch.nn.ConvTranspose2d(in_channels=self.m.out_channels, out_channels=self.m.in_channels,
                                            kernel_size=self.m.kernel_size, stride=self.m.stride, padding=self.m.padding, 
                                            output_padding=(0,0),groups=self.m.groups,bias=None) #bias=None


        # self.m.weight (c_out,c_in, k0,k1)
        # self.inv_m.weight (c_in, c_out, k0,k1)
        self.inv_m.weight=nn.Parameter(self.m.weight) # no need to do transpose(0,1)

    def cal_outpad(self):
        Z = self.m.forward(self.m.X).detach()

        output_padding = self.m.X.size()[2] - (
                (Z.size()[2] - 1) * self.m.stride[0] - 2 * self.m.padding[0] + self.m.kernel_size[0])
        return output_padding


    def gradprop2(self, DY, weight):
        # Z = self.m.forward(self.m.X)

        # output_padding = self.m.X.size()[2] - (
        #         (Z.size()[2] - 1) * self.m.stride[0] - 2 * self.m.padding[0] + self.m.kernel_size[0])
        output_padding=self.cal_outpad()

        return F.conv_transpose2d(DY, weight, stride=self.m.stride, padding=self.m.padding, output_padding=output_padding)

    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
            
        if self.me_eff:
            self.m.X=self.m.X.detach()
            
        def final_backward(R_p, pw, nw, X1):
            X = X1
            L = X * 0 + \
                torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = X * 0 + \
                torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.m.weight, bias=None, stride=self.m.stride, padding=self.m.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.m.stride, padding=self.m.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.m.stride, padding=self.m.padding)
            
            if self.me_eff:
                Sp = safe_divide(R_p, Za.detach())
            else:
                Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.m.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp

        pw = torch.clamp(self.m.weight, min=0)
        nw = torch.clamp(self.m.weight, max=0)

        if self.m.X.shape[1] == 3:
            return final_backward(cam, pw, nw, self.m.X), None

        # Compute - M
        #Y = self.m.forward(self.m.X)
        # S = grad_outputs[0]
        # S = grad_outputs
        # grad = torch.autograd.grad(Y, self.m.X, S,retain_graph=True)
        grad=F.conv_transpose2d(grad_outputs,self.m.weight,stride=self.m.stride,padding=self.m.padding,output_padding=self.cal_outpad())

        # M = F.relu((self.m.X * grad[0]).mean(dim=1, keepdim=True).expand_as(self.m.X))
        M = F.relu((self.m.X * grad).mean(dim=1, keepdim=True).expand_as(self.m.X))
        M = safe_divide(M, minmax_dims(M, 'max'))

        # Type Grad
        #Y = self.m.forward(self.m.X) * cam.ne(0).type(cam.type())
        # S = grad_outputs[0]
        # S = grad_outputs
        # grad = torch.autograd.grad(Y, self.m.X, S,retain_graph=True)
        grad=F.conv_transpose2d(grad_outputs,self.m.weight,stride=self.m.stride,padding=self.m.padding,output_padding=self.cal_outpad())

        # gradcam = self.m.X * F.adaptive_avg_pool2d(grad[0], 1)
        gradcam = self.m.X * F.adaptive_avg_pool2d(grad, 1)
        gradcam = gradcam.mean(dim=1, keepdim=True).expand_as(self.m.X)

        # Compute - C
        xabs = self.m.X if torch.all(self.m.X>=0) else self.m.X.abs()
        wabs = self.m.weight.abs()
        Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * cam.ne(0).type(cam.type())

        S = safe_divide(cam, Zabs)
        # grad = torch.autograd.grad(Zabs, xabs, S,retain_graph=True)
        grad=F.conv_transpose2d(S,wabs,stride=self.m.stride, padding=self.m.padding,output_padding=self.cal_outpad())
        # C = xabs * grad[0]
        C = xabs * grad

        # Compute Factorization - F_x, F_dx
        # Y = self.m.forward(self.m.X) * cam.ne(0).type(cam.type())
        # S = grad_outputs[0]
        #S = grad_outputs
        # grad = torch.autograd.grad(Y, self.m.X, S,retain_graph=True)
        grad=F.conv_transpose2d(grad_outputs* cam.ne(0).type(cam.type()),self.m.weight,stride=self.m.stride,padding=self.m.padding,output_padding=self.cal_outpad())

        # F_x, F_dx = get_factorization(self.m.X, grad[0], C.mean(dim=1, keepdim=True))
        F_x, F_dx = get_factorization(self.m.X, grad, C.mean(dim=1, keepdim=True))

        F_x = F.relu(F_x)
        F_dx = F.relu(F_dx)

        # Compute - A
        wabs = self.m.weight.abs()
        xabs = torch.ones_like(self.m.X)
        xabs.requires_grad_() #TODO check
        Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * cam.ne(0).type(cam.type())
        if self.me_eff:
            S = safe_divide(cam, Zabs.detach())
        else:
            S = safe_divide(cam, Zabs)
        # grad = torch.autograd.grad(Zabs, xabs, S,retain_graph=True)
        grad = F.conv_transpose2d(S,wabs,stride=self.m.stride,padding=self.m.padding,output_padding=self.cal_outpad())
        # A = xabs * grad[0]
        A = xabs * grad

        # Compute residual - R
        R = 0

        numer = 0
        if "no_fx" not in kwargs.keys() or not kwargs["no_fx"]:
            numer += F_x

        if "no_m" not in kwargs.keys() or not kwargs["no_m"]:
            numer += M

        if "gradcam" in kwargs.keys() and kwargs["gradcam"]:
            numer = F.relu(gradcam)

        if "no_reg" not in kwargs.keys() or not kwargs["no_reg"]:
            R += safe_divide(numer, 1 + torch.exp(-C.detach()))
        else:
            R += numer

        if "no_fdx" not in kwargs.keys() or not kwargs["no_fdx"]:
            R += F_dx

        if "no_a" not in kwargs.keys() or not kwargs["no_a"]:
            R += A

        R = R * C.ne(0).type(C.type())

        # Delta shift
        result = delta_shift(C, R)

        if "flat" in kwargs.keys() and kwargs["flat"]:
            cam_nonzero = cam.ne(0).type(cam.type())
            xabs = self.m.X if torch.all(self.m.X>=0) else self.m.X.abs()
            wabs = self.m.weight.abs()
            Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * cam_nonzero
            if self.me_eff:
                S = safe_divide(cam, Zabs.detach())
            else:   
                S = safe_divide(cam, Zabs)
            # result = xabs * torch.autograd.grad(Zabs, xabs, S,retain_graph=True)[0]
            result=xabs*F.conv_transpose2d(S,wabs,stride=self.m.stride,padding=self.m.padding,output_padding=self.cal_outpad())

        # Gradient
        # Y = self.m.forward(self.m.X)
        S = grad_outputs
        # grad_out = torch.autograd.grad(Y, self.m.X, S)
        # grad_out = torch.autograd.grad(Y, self.m.X, S,retain_graph=True)[0]
        grad_out=F.conv_transpose2d(S,self.m.weight,stride=self.m.stride,padding=self.m.padding,output_padding=self.cal_outpad())
        return result, grad_out


