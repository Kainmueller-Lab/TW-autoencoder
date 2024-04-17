import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [ 'Clone', 'Add', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide']


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())



def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()


    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C



    def RAP_relprop(self, R_p):
        return R_p


class RelPropSimple(RelProp):
 
    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = self.m.forward(self.m.in_tensor)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.m.in_tensor, Sp)[0]
            if torch.is_tensor(self.m.in_tensor) == False:
                Rp = []
                Rp.append(self.m.in_tensor[0] * Cp)
                Rp.append(self.m.in_tensor[1] * Cp)
            else:
                Rp = self.m.in_tensor * (Cp)
            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


# --------------------------------------
# Layers for LRP
# --------------------------------------
class RAP_relu( RelProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.MaxPool2d), "Please tie with a relu layer"
        self.m=m

    def forward(self, R):
        return self.RAP_relprop(R)



class RAP_dropout( RelProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Dropout), "Please tie with a dropout layer"
        self.m=m

    def forward(self, R):
        return self.RAP_relprop(R)



class RAP_maxpool2d(RelPropSimple):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.MaxPool2d), "Please tie with a maxpool2d layer"
        self.m=m
    
    def forward(self, R):
        return self.RAP_relprop(R)



class RAP_avgpool2d(RelPropSimple):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.AdaptiveAvgPool2d), "Please tie with an adaptive avgpool2d layer"
        self.m=m

    def forward(self, R):
        return self.RAP_relprop(R)

# class AvgPool2d(nn.AvgPool2d, RelPropSimple):
#     pass

# TODO check how to deal with this layer, especially in resnet
class RAP_add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)


class RAP_clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs



    def RAP_relprop(self, R_p):
        def backward(R_p):
            Z = []
            for _ in range(self.num):
                Z.append(self.X)

            Spp = []
            Spn = []

            for z, rp, rn in zip(Z, R_p):
                Spp.append(safe_divide(torch.clamp(rp, min=0), z))
                Spn.append(safe_divide(torch.clamp(rp, max=0), z))

            Cpp = self.gradprop(Z, self.X, Spp)[0]
            Cpn = self.gradprop(Z, self.X, Spn)[0]

            Rp = self.X * (Cpp * Cpn)

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class RAP_cat(RelProp):
    # def forward(self, inputs, dim):
    #     self.__setattr__('dim', dim)
    #     return torch.cat(inputs, dim)


    def forward(self, R_p):
        def backward(R_p):
            Z = self.forward(self.X, self.dim)
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, self.X, Sp)

            Rp = []

            for x, cp in zip(self.X, Cp):
                Rp.append(x * (cp))

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


# class Sequential(nn.Sequential):
#     def relprop(self, R, alpha):
#         for m in reversed(self._modules.values()):
#             R = m.relprop(R, alpha)
#         return R

#     def RAP_relprop(self, Rp):
#         for m in reversed(self._modules.values()):
#             Rp = m.RAP_relprop(Rp)
#         return Rp


class RAP_BN2d( RelProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.BatchNorm2d), "Please tie with a batchnorm2d layer"
        self.m=m

    

    def forward(self, R_p):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1

        def backward(R_p):
            X = self.m.X

            weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.m.eps).pow(0.5))

            if torch.is_tensor(self.m.bias):
                bias = self.m.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(bias * R_p.ne(0).type(self.m.bias.type()),
                                     R_p.ne(0).type(self.m.bias.type()).sum(dim=[2, 3], keepdim=True))
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if torch.is_tensor(self.m.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class RAP_linear( RelProp):
    def __init__(self,m,**kwargs):
        super().__init__()
        assert isinstance(m,nn.Linear), "Please tie with a linear layer"
        self.m=m

        # should be deleted later
        self.inv_m=nn.Linear(in_features=self.m.out_features, out_features=self.m.in_features, bias=None) #bias=None
        self.inv_m.weight=nn.Parameter(self.m.weight.t())


    def forward(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n

            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=-1, keepdim=True) - R.sum(dim=-1, keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.linear(x1, w1) * R_nonzero
            Za2 = - F.linear(x1, w2) * R_nonzero

            Zb1 = - F.linear(x2, w1) * R_nonzero
            Zb2 = F.linear(x2, w2) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)

            return C1 + C2

        def first_prop(pd, px, nx, pw, nw):
            Rpp = F.linear(px, pw) * pd
            Rpn = F.linear(px, nw) * pd
            Rnp = F.linear(nx, pw) * pd
            Rnn = F.linear(nx, nw) * pd
            Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
            Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

            Z1 = F.linear(px, pw)
            Z2 = F.linear(px, nw)
            Z3 = F.linear(nx, pw)
            Z4 = F.linear(nx, nw)

            S1 = safe_divide(Rpp, Z1)
            S2 = safe_divide(Rpn, Z2)
            S3 = safe_divide(Rnp, Z3)
            S4 = safe_divide(Rnn, Z4)
            C1 = px * self.gradprop(Z1, px, S1)[0]
            C2 = px * self.gradprop(Z2, px, S2)[0]
            C3 = nx * self.gradprop(Z3, nx, S3)[0]
            C4 = nx * self.gradprop(Z4, nx, S4)[0]
            bp = self.bias * pd * safe_divide(Pos, Pos + Neg)
            bn = self.bias * pd * safe_divide(Neg, Pos + Neg)
            Sb1 = safe_divide(bp, Z1)
            Sb2 = safe_divide(bn, Z2)
            Cb1 = px * self.gradprop(Z1, px, Sb1)[0]
            Cb2 = px * self.gradprop(Z2, px, Sb2)[0]
            return C1 + C4 + Cb1 + C2 + C3 + Cb2

        def backward(R_p, px, nx, pw, nw):
            Rp = f(R_p, pw, nw, px, nx)
            return Rp

        def redistribute(Rp_tmp):
            Rp = torch.clamp(Rp_tmp, min=0)
            Rn = torch.clamp(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3

        pw = torch.clamp(self.m.weight, min=0)
        nw = torch.clamp(self.m.weight, max=0)
        X = self.m.X
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  ## first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A = redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        return Rp


class RAP_transposeconv2d( RelProp):
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

    

    def forward(self, R_p):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(R_nonzero, dim=[1, 2, 3], keepdim=True)) * torch.ne(R, 0).type(
                R.type())
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
            R_pos = torch.clamp(R, min=0)
            R_neg = torch.clamp(R, max=0)
            S1 = safe_divide((R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
            C1 = x1 * self.gradprop(Za1, x1, S1)[0]
            S1n = safe_divide((R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
            C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
            S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
            C2 = x1 * self.gradprop(Za2, x1, S2)[0]
            S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
            C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
            Cp = C1 + C2
            Cn = C2n + C1n
            C = (Cp + Cn)
            C = shift_rel(C, C.sum(dim=[1, 2, 3], keepdim=True) - R.sum(dim=[1, 2, 3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            if w1.shape[2] == 1: #TODO this part of code is not coming from RAP author
                xabs = self.m.X.abs()
                wabs = self.m.weight.abs()
                Zabs = F.conv2d(xabs, wabs, bias=None, stride=self.m.stride, padding=self.m.padding) * R_nonzero
                S = safe_divide(R, Zabs)
                C = xabs * self.gradprop(Zabs, xabs, S)[0]

                return C
            else:
                R_nonzero = R.ne(0).type(R.type())
                Za1 = F.conv2d(x1, w1, bias=None, stride=self.m.stride, padding=self.m.padding) * R_nonzero
                Za2 = - F.conv2d(x1, w2, bias=None, stride=self.m.stride, padding=self.m.padding) * R_nonzero

                Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.m.stride, padding=self.m.padding) * R_nonzero
                Zb2 = F.conv2d(x2, w2, bias=None, stride=self.m.stride, padding=self.m.padding) * R_nonzero

                C1 = pos_prop(R, Za1, Za2, x1)
                C2 = pos_prop(R, Zb1, Zb2, x2)

                return C1 + C2

        def backward(R_p, px, nx, pw, nw):
            Rp = f(R_p, pw, nw, px, nx)
            return Rp

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
        px = torch.clamp(self.m.X, min=0)
        nx = torch.clamp(self.m.X, max=0)
  
        if self.m.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.m.X)
         
        else:
            Rp = backward(R_p, px, nx, pw, nw)
        return Rp


