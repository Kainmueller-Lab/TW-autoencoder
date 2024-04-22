from ._init_pretrainmodel import initialize_pretrainmodel
from ._hookfn import FwdHooks
from .decoder import Tied_weighted_decoder
import torch
from torch import nn
from torch.autograd import Variable
from typing import  Optional
import sys
class Unrolled_lrp(nn.Module):
    

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        xai: str = "LRP_epsilon",
        epsilon: float = 1e-8,
        alpha: float = 1.0,
        detach_bias: bool = True,
        input_size: tuple = (512,512),
        ablation_test: bool = False,
        normal_relu : bool = False,
        normal_deconv :bool = False,
        normal_unpool : bool = False,
        multiply_input : bool = False,
        remove_heaviside : bool = False,
        remove_last_relu : bool = False,
        add_bottle_conv : bool = False

    ):
        super().__init__()

        self.num_segments=classes
        # the encoder of the unrolled_lrp model is directly initialized with the following functions
        self.encoder = initialize_pretrainmodel(encoder_name, classes, in_channels, feature_extract=False, use_pretrained=True) 
        self.fwd_hooks=FwdHooks(xai)
        self.register_hooks(self.encoder)

        
        # run test_forward for assigning the attribute of in_tensor.shape 
        # for the transposeconv_2d layer in the decoder
        # for adapative avgpool layer to set initialize the const weight 
        self.test_forward(self.encoder,in_channels, input_size)


        self.ablation_test=ablation_test
        self.tied_weights_decoder= Tied_weighted_decoder(encoder_name,
            self.encoder, xai, epsilon, alpha,
            detach_bias=detach_bias,
            ablation_test=ablation_test,
            normal_relu=normal_relu, normal_deconv=normal_deconv,
            normal_unpool=normal_unpool,
            multiply_input=multiply_input,remove_heaviside=remove_heaviside,
            remove_last_relu=remove_last_relu, add_bottle_conv=add_bottle_conv)

        self.normal_deconv=normal_deconv
        
    def forward(self, x, targets=None, only_classification=False):
        '''
        targets: torch.tensor (N,)
        '''
        if only_classification:
            z = self.encoder(x) 
            return z
        else:
            if self.normal_deconv: # only one decoder branch
                z = self.encoder(x) 
                heatmaps=self.tied_weights_decoder(z,0) # random put a class_idx
            else:
                z = self.encoder(x) 
                # calculate the heatmap for all classes, self.num_segments decoder branches
                heatmaps_poket=[]
                for i in range(self.num_segments):
                    tmp_heatmaps=self.tied_weights_decoder(self.indi_features(z,i),i)
                    heatmaps_poket.append(tmp_heatmaps) # (N, H, W)
                heatmaps=torch.stack(heatmaps_poket,dim=1)  # (N, num_segments, H, W)
            return z, heatmaps


    def forward_decoder_detached(self, z):  
        z = z.clone() # safety opperation
        z = z.detach() # NOTE delete not needed computational graph of z for this computation
        heatmaps_poket=[]

        for i in range(self.num_segments):
            tmp_heatmaps=self.tied_weights_decoder(self.indi_features(z,i),i)
            heatmaps_poket.append(tmp_heatmaps.detach()) # (N, H, W) NOTE here the computational graph of every tmp_heatmap gets deleted
        heatmaps=torch.stack(heatmaps_poket,dim=1)  # (N, num_segments, H, W)
        
        return heatmaps


    def forward_individual_class(self, z, class_idx):  
        heatmap=self.tied_weights_decoder(self.indi_features(z,class_idx),class_idx) # (N, H, W)  
        return heatmap


    # The usage of the testfoward:
    # for the transposeconv_2d layer in the decoder
    # for adapative avgpool layer to set initialize the const weight 
    def test_forward(self,module,in_channels,input_size):
        x=Variable(torch.ones(2, in_channels, *input_size)) # bacth_size must larger than 1 for batchnorm1d   
        y=module(x)     
        return


    def register_hooks(self, parent_module):
        
        for mod in parent_module.children():
            # print("LRP processing module... ", mod)     
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(
                self.fwd_hooks.get_layer_fwd_hook(mod))


    def indi_features(self,z,targets):
        assert z.shape[-1]==self.num_segments, "The encoder part has problem."
        if targets==None:
            tmp_list=[]         
            for i in range(self.num_segments):
                tmp_list.append(self.initialise_rel("normal",z,i))
            return torch.stack(tmp_list,dim=0)
        else:
            return self.initialise_rel("normal",z,targets)
            

    def initialise_rel(self,init_type,class_scores,targets):
        device=class_scores.device
        N=class_scores.shape[0]
        
        if init_type == "softmax":
            T=self.softmaxlayer_lrp(class_scores,targets,device)

        elif init_type == "normal":
            T = torch.zeros(class_scores.shape).to(device)
            T[range(N),targets]=class_scores[range(N),targets]


        elif init_type== "contrastive":
            T=torch.abs(class_scores)
            T[range(N),targets]=0.0

        elif init_type == "normal-contrastive":
            T = torch.ones(class_scores.shape).to(device)*(-1)
            T[range(N),targets]=1.0
            T=T*class_scores
  
        else:
            raise NotImplementedError

        return T


    def softmaxlayer_lrp(class_score,targets,device):
        '''
        Input:
            class_score: torch tensor, shape [N, num_segments], should be the logit
            targets: list of GT labels, len=N or torch.tensor with shape (N,)
            device: cpu or cuda
        Return:
            R, the relevance distribution, torch tensor, shape [N, num_segments]

        example:
            R1= Z1, target class
            R2=-Z2*exp(-(Z1-Z2))/(exp(-(Z1-Z2))+exp(-(Z1-Z3)))
            R3=-Z3*exp(-(Z1-Z3))/(exp(-(Z1-Z2))+exp(-(Z1-Z3)))
            
        principle:
            assume c is the GT label or intertesed class label
            Rc=Zc
            Rc'=-Zc'*exp(-(Zc-Zc'))/sum_{c''!=c}exp(-(Zc-Zc''))
            
        '''

        # -(Zc-Zc'')
        assert class_score.dim()==2,"The dimension of the class_score tensor shouold be 2." 
        # targets=torch.LongTensor(targets).to(device)
        targets=torch.LongTensor(targets)
        scores_diff=class_score-class_score.gather(1,targets.view(-1,1))

        # exp(Zc'-Zc)/sum_{c''!=c}exp(-(Zc-Zc''))
        R=torch.exp(scores_diff)/(torch.exp(scores_diff).sum(dim=1,keepdim=True)-1.0)

        # Rc'=-Zc'*exp(Zc'-Zc)/sum_{c''!=c}exp(-(Zc-Zc'')) &  Rc=Zc
        R=(-1.0)*class_score*R
        R[range(R.shape[0]),targets]=class_score[range(R.shape[0]),targets]

        return R

        
    


