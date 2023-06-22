#!/usr/bin/python
# coding: utf-8

 

from train_resnet import *

 

#class Model(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.net = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=2)
#        n_features = self.net._fc.in_features
#        self.net._fc = nn.Linear(in_features=n_features, out_features=2, bias=True)
    
#    def forward(self, x):
#        out = self.net(x)
#        return out
    
if __name__ == "__main__":
   
    
    set_of_files = ["Flair_res111.nii.gz"]
    norm_set_of_files = ["Flair_res111/Flair_res111_trans.nii.gz"]
            #"T1_res111/T1_res111_trans.nii.gz",
            #"T13D_res111/T13D_res111_trans.nii.gz",
            #"T2_res111/T2_res111_trans.nii.gz")
    #modelfile = "/data/data/T1-T2-resnet_min_loss_model_classification3d_array.pth"#train_all_type(df_train, df_valid, mri_type="Flair-T2")#"all"
    #print(modelfile)
    #a = load_image_3d(scan_id="Pcnls_1")
    #medCAM(a, modelfile, mri_types = ("Flair", "T2")) #"T1", "T13D", "T2"))
   
    train_loss, valid_ROC, modelfile1,  modelfile3 = train_all_type(norm_set_of_files=norm_set_of_files,random_state=random_state,mri_type=["Flair"])  # -T2")"all"

    print(train_loss, valid_ROC, modelfile1,  modelfile3)


    
#ROC = 0.23809523809523808
#auc = 0.5
#precision = 0.3125
#recall = 0.35714285714285715
#fscore = 0.3333333333333333


    
#a = np.array(a)
#a = torch.tensor(a, dtype=torch.float32)
#print(a.shape)
#print(a[0].shape)
#print(a.squeeze(0).shape)
#show_plt(a[0], "Flair")
#b = a.unsqueeze(0)
#print(b.shape)








# In[ ]:




 
#a = load_image_3d(scan_id="Pcnls_1")
#print(a[0].shape)
#print(np.min(a), np.max(a), np.mean(a), np.median(a))
    


