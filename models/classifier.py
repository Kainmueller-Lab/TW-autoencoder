from torchvision.models import vgg16
import torch


# --------------------------------------
# Main function for training and test
# --------------------------------------
class classifier(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        
        #self.model = Network(args)
        self.model = vgg16()
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.beta=1 #hyperparam for classification loss

        def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = False

        model_ft = vgg16(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.features[0] = nn.Conv2d(img_channel, 64, 3, 1, 1)









        self.only_bg=False
        self.loss_mode = 'all'
        self.file_name=f"only_bg_alpha_{self.alpha}_beta_{self.beta}" if self.only_bg==True else f"alpha_{self.alpha}_beta_{self.beta}"

        # heatmap save folder
        self.heatmap_save_folder=f"results/heatmaps/{self.args.dataset}_{self.args.num_classes}/{self.file_name}/"
        os.makedirs(self.heatmap_save_folder,exist_ok=True)
        # tensorboard writers
        os.makedirs(f"runs/{self.args.dataset}",exist_ok=True)
        run_name1=f"runs/{self.args.dataset}_{self.args.num_classes}/{self.file_name}_train"
        run_name2=f"runs/{self.args.dataset}_{self.args.num_classes}/{self.file_name}_val"
        self.tb_writer1=SummaryWriter(run_name1)# for train_data
        self.tb_writer2=SummaryWriter(run_name2)# for val_data
  

    def _init_dataset(self):
        if self.args.dataset == 'MNIST':
            self.data = MNIST(self.args)
        elif self.args.dataset == 'EMNIST':
            self.data = EMNIST(self.args)
        elif self.args.dataset == 'FashionMNIST':
            self.data = FashionMNIST(self.args)
        elif self.args.dataset== 'Lizard':
            self.data=Lizard(self.args)
        elif self.args.dataset == 'PASCAL':
            self.data = PASCAL(self.args)
        else:
            print("Dataset not supported")


    def generate_class_weights(self, sem_gts):
        weights = torch.stack([ torch.sum(sem_gts==i,axis=(1,2)) for i in range(self.args.num_classes) ],dim=1) # TODO readjust to 7 classes -> done
        weights=1/torch.sum(weights,dim=0)
        weights[weights == float('inf')] = 0
        return weights


    def add_bg_heatmap(self,heatmaps):
        if self.args.num_classes==6:
            N,_,H,W=heatmaps.shape
            bg=torch.zeros((N,1,H,W)).to(self.device)
            heatmaps=torch.cat((bg, heatmaps),dim=1) #(N, 7, H, W)  

        return heatmaps    


    def loss_function(self, pred_class_scores, class_gt, heatmaps,seg_gt):
        
        # heatmap loss
        heatmap_loss=0

        if self.only_bg:
            L_bg=bg_fg_prob_loss(pred_class_scores, class_gt, heatmaps,seg_gt)
            L_duc=depress_unexist_class_loss(pred_class_scores, class_gt, heatmaps,seg_gt)
            heatmap_loss=L_bg+L_duc
            
        else:
            if self.loss_mode == 'min':
                weights = (1-torch.concat((torch.ones((class_gt.size(0),1)).to(self.device),class_gt),dim=1))
                heatmap_loss=nn.CrossEntropyLoss(weight=weights,reduction="sum")(input=heatmaps,target=0*seg_gt)
            else:
                weights=self.generate_class_weights(seg_gt)
                heatmap_loss=nn.CrossEntropyLoss(reduction="mean")(input=heatmaps,target=seg_gt) # TODO check if seg_gt right format, introduce back weights

        # classification loss
        classification_loss=torch.nn.BCEWithLogitsLoss(reduction="sum")(pred_class_scores,class_gt.float()) # size (N, num_classes) flat tensor

        # total loss
        t_loss=self.beta*classification_loss+self.alpha*heatmap_loss
        return t_loss, self.alpha*heatmap_loss, self.beta*classification_loss


    def train(self, epoch):
        self.model.train()
        train_loss = 0
        metrics=init_metrics(self.args.num_classes)
        # 16.11.2022 rewrote
        for batch_idx, (data, class_labels,sem_gts) in enumerate(self.train_loader):
           
            # generate sem_gt from data
            sem_gts=sem_gts.long().to(self.device)
            class_labels=class_labels.to(self.device)
            data = data.to(self.device)

            self.optimizer.zero_grad()
            class_scores,heatmaps = self.model(data,class_labels)
            heatmaps=self.add_bg_heatmap(heatmaps) 
            loss, loss1,loss2= self.loss_function(class_scores,class_labels,heatmaps,sem_gts)
  
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            metrics=get_accuracy("multi-label-classification", metrics, class_scores,class_labels)
            metrics=get_accuracy("multi-label-segmentation", metrics, heatmaps,sem_gts)

            if self.args.wandb != 'None':
                wandb_utils.log_training(epoch,batch_idx,len(self.train_loader),len(data),
                    loss2.item(),loss1.item(),metrics,
                    data,sem_gts,heatmaps,
                    class_labels,class_scores) 
            
            if batch_idx==300:
                print(f"====> Epoch: {epoch} Train (random batch): heatmap loss {loss1.item()/len(data):.10f}; classification loss {loss2.item()/len(data):.4f}")
                save_raw_heatmp_sem_gt(4,2,data,heatmaps,sem_gts, f"{epoch}_train_heatmap.png" , self.heatmap_save_folder)

        gen_log_message("train",epoch,train_loss / len(self.train_loader.dataset), metrics,self.args.num_classes)
        write_summaries(self.tb_writer1, train_loss / len(self.train_loader.dataset), metrics,self.args.num_classes, epoch+1)


    def test(self, epoch):
        
        self.model.eval()
        test_loss = 0
        metrics=init_metrics(self.args.num_classes)
        
        # set list of logging images
        log_batch = wandb_utils.fixed_random_examples(len(self.test_loader),25)
        log_examples = None
        log_gt_heatmaps = None
        log_pred_heatmaps = None

        with torch.no_grad():
            for i, (data, class_labels,sem_gts) in enumerate(self.test_loader):
                sem_gts=sem_gts.long().to(self.device)
       
                data = data.to(self.device)
                class_labels=class_labels.to(self.device)
                class_scores,heatmaps = self.model(data,class_labels)
                heatmaps=self.add_bg_heatmap(heatmaps) 
                tloss, test_loss1, test_loss2 = self.loss_function(class_scores, class_labels,heatmaps,sem_gts)
                test_loss += tloss.item()
              
                metrics=get_accuracy("multi-label-classification",metrics, class_scores,class_labels)
                metrics=get_accuracy("multi-label-segmentation",metrics,heatmaps,sem_gts)

                # save images and heatmaps for logging
                if self.args.wandb != 'None':
                    log_examples,log_gt_heatmaps,log_pred_heatmaps = wandb_utils.storing_imgs(i,log_batch,
                                                                                                data,sem_gts,heatmaps,
                                                                                                log_examples,log_gt_heatmaps,log_pred_heatmaps)     

                #save heatmaps
                if i==30:
                    print(f"      Test (random batch): test H-L {test_loss1.item()/len(data):.10f}; test C-L {test_loss2.item()/len(data):.4f}")
                    save_raw_heatmp_sem_gt(4,2,data,heatmaps,sem_gts, f"{epoch}_test_heatmap.png" , self.heatmap_save_folder)
                   
        test_loss /= len(self.test_loader.dataset)

        gen_log_message("test",epoch,test_loss, metrics,self.args.num_classes)
        avg_segmentation_accu,avg_classification_accu = write_summaries(self.tb_writer2,test_loss, metrics,self.args.num_classes, epoch+1,return_values=True)

        if self.args.wandb != 'None':
            wandb_utils.log_testing(epoch,len(self.train_loader),self.args.batch_size,
                                    test_loss,avg_segmentation_accu,avg_classification_accu,
                                    log_examples,log_gt_heatmaps,log_pred_heatmaps)  