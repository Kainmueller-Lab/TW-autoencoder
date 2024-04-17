import torch
import torch.nn as nn
import torch.nn.functional as F
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
        Y = m.forward(m.in_tensor)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, m.in_tensor, S)

        return cam, grad_out


class AGFPropSimple(AGFProp):
    def AGF(self, m, cam=None, grad_outputs=None, **kwargs):
        def backward(cam):
            Z = m.forward(m.in_tensor) * cam.ne(0).type(cam.type())
            S = safe_divide(cam, Z)

            if torch.is_tensor(m.in_tensor) == False:
                # TODO: check
                result = []
                grad = torch.autograd.grad(Z, m.in_tensor, S)
                result.append(m.in_tensor[0] * grad[0])
                result.append(m.in_tensor[1] * grad[1])
            else:
                grad = torch.autograd.grad(Z, m.in_tensor, S)
                result = m.in_tensor * grad[0]
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
        Y = m.forward(m.in_tensor)
        S = grad_outputs
        #grad_out = torch.autograd.grad(Y, m.X, S)
        grad_out = torch.autograd.grad(Y, m.in_tensor, S)[0]

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

        C = torch.autograd.grad(Z, self.X, S)

        R = self.X * C[0]

        # Gradient
        Y = self.forward(self.X, self.num)
        S = grad_outputs
        grad_out = torch.autograd.grad(Y, self.X, S)

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
        assert isinstance(m,nn.MaxPool2d), "Please tie with a relu layer"
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
    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
        return self.AGF(self.m, cam, grad_outputs, **kwargs)


class AGF_dropout(AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Dropout), "Please tie with a dropout layer"
        self.m=m

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

    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
        return self.AGF(self.m, cam, grad_outputs, **kwargs)
   


# class AvgPool2d(nn.AvgPool2d, AGFPropSimple):
#     pass


class AGF_linear( AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Linear), "Please tie with a linear layer"
        self.m=m
        
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
            Y = F.softmax(yt * torch.exp(-0.5 * ((Y - yt) / sigma) ** 2), dim=1)

            # Gradients stream
            grad_out = torch.autograd.grad(Y, self.m.X, tgt)[0]
            # grad_out = torch.autograd.grad(Y, self.m.X, tgt) #TODO check

            #result = self.m.X * grad_out[0] #TODO check
            result = self.m.X * grad_out
        else:
            # Compute - C
            xabs = self.m.X.abs()
            wabs = self.m.weight.abs()
            Zabs = F.linear(xabs, wabs) * cam.ne(0).type(cam.type())

            S = safe_divide(cam, Zabs)
            grad = torch.autograd.grad(Zabs, xabs, S)
            C = xabs * grad[0]

            # Compute - M
            Y = self.m.forward(self.m.X)
            # S = grad_outputs[0]
            S = grad_outputs #TODO check

          
            grad = torch.autograd.grad(Y, self.m.X, S)
            if self.m.reshape_gfn is not None:
                x = self.m.reshape_gfn(self.m.X)
                g = self.m.reshape_gfn(grad[0])
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
            Y = self.m.forward(self.m.X)
            S = grad_outputs
            # grad_out = torch.autograd.grad(Y, self.m.X, S)
            grad_out = torch.autograd.grad(Y, self.m.X, S)[0]

 
        return result, grad_out


class AGF_transposeconv2d( AGFProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Conv2d), "Please tie with a conv2d layer"
        self.m=m

        # should be deleted later
        self.inv_m=torch.nn.ConvTranspose2d(in_channels=self.m.out_channels, out_channels=self.m.in_channels,
                                            kernel_size=self.m.kernel_size, stride=self.m.stride, padding=self.m.padding, 
                                            output_padding=(0,0),groups=self.m.groups,bias=None) #bias=None


        # self.m.weight (c_out,c_in, k0,k1)
        # self.inv_m.weight (c_in, c_out, k0,k1)
        self.inv_m.weight=nn.Parameter(self.m.weight) # no need to do transpose(0,1)


    def gradprop2(self, DY, weight):
        Z = self.m.forward(self.m.X)

        output_padding = self.m.X.size()[2] - (
                (Z.size()[2] - 1) * self.m.stride[0] - 2 * self.m.padding[0] + self.m.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.m.stride, padding=self.m.padding, output_padding=output_padding)

    def forward(self, cam=None, grad_outputs=None, **kwargs):
        if isinstance(cam, tuple) and len(cam)==2:
            grad_outputs=cam[1]
            cam=cam[0]
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

            Sp = safe_divide(R_p, Za)

            Rp = X * self.gradprop2(Sp, self.m.weight) - L * self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp

        pw = torch.clamp(self.m.weight, min=0)
        nw = torch.clamp(self.m.weight, max=0)

        if self.m.X.shape[1] == 3:
            return final_backward(cam, pw, nw, self.m.X), None

        # Compute - M
        Y = self.m.forward(self.m.X)
        # S = grad_outputs[0]
        S = grad_outputs
        grad = torch.autograd.grad(Y, self.m.X, S)

        M = F.relu((self.m.X * grad[0]).mean(dim=1, keepdim=True).expand_as(self.m.X))
        M = safe_divide(M, minmax_dims(M, 'max'))

        # Type Grad
        Y = self.m.forward(self.m.X) * cam.ne(0).type(cam.type())
        # S = grad_outputs[0]
        S = grad_outputs
        grad = torch.autograd.grad(Y, self.m.X, S)

        gradcam = self.m.X * F.adaptive_avg_pool2d(grad[0], 1)
        gradcam = gradcam.mean(dim=1, keepdim=True).expand_as(self.m.X)

        # Compute - C
        xabs = self.m.X.abs()
        wabs = self.m.weight.abs()
        Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * cam.ne(0).type(cam.type())

        S = safe_divide(cam, Zabs)
        grad = torch.autograd.grad(Zabs, xabs, S)
        C = xabs * grad[0]

        # Compute Factorization - F_x, F_dx
        Y = self.m.forward(self.m.X) * cam.ne(0).type(cam.type())
        # S = grad_outputs[0]
        S = grad_outputs
        grad = torch.autograd.grad(Y, self.m.X, S)

        F_x, F_dx = get_factorization(self.m.X, grad[0], C.mean(dim=1, keepdim=True))

        F_x = F.relu(F_x)
        F_dx = F.relu(F_dx)

        # Compute - A
        wabs = self.m.weight.abs()
        xabs = torch.ones_like(self.m.X)
        xabs.requires_grad_()
        Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * cam.ne(0).type(cam.type())
        S = safe_divide(cam, Zabs)
        grad = torch.autograd.grad(Zabs, xabs, S)
        A = xabs * grad[0]

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
            R += safe_divide(numer, 1 + torch.exp(-C))
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
            xabs = self.m.X.abs()
            wabs = self.m.weight.abs()
            Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * cam_nonzero
            S = safe_divide(cam, Zabs)
            result = xabs * torch.autograd.grad(Zabs, xabs, S)[0]

        # Gradient
        Y = self.m.forward(self.m.X)
        S = grad_outputs
        # grad_out = torch.autograd.grad(Y, self.m.X, S)
        grad_out = torch.autograd.grad(Y, self.m.X, S)[0]

        return result, grad_out


