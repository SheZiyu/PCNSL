#!/usr/bin/python
# coding: utf-8



# In[ ]:

import train_resnet
from train_resnet import *


# In[ ]:


rows, cols = 1, 3

def show_plt_(img, name):
    plt.figure(figsize=(15,15), dpi=80)
    plt.figure(1)

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)
    
    x = img.shape[0] // 2
    y = 68
    z = 58
    
    plt.sca(ax1)
    plt.imshow(img[x, :, :], cmap="Spectral")
     
    plt.sca(ax2)
    plt.imshow(img[:, y, :], cmap="Spectral")
    
    plt.sca(ax3)
    plt.imshow(img[:, :, z], cmap="Spectral")
    plt.colorbar()
    
    plt.savefig("{}".format(name))


    plt.show()
    plt.figure(1).clear()


rows, cols = 1, 3


def show_plt_old(img, name):
    plt.figure(figsize=(15, 15), dpi=80)
    plt.figure(1)

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)

    x = img.shape[0] // 2
    y = 68
    z = 58


    plt.sca(ax1)
    plt.imshow(img[x, :, :], cmap="gray")

    plt.sca(ax2)
    plt.imshow(img[:, y, :], cmap="gray")

    plt.sca(ax3)
    plt.imshow(img[:, :, z], cmap="gray")
    #plt.colorbar()

    plt.savefig("{}".format(name))

    plt.show()
    plt.figure(1).clear()


# In[ ]:


#a = load_image_3d(scan_id="Pcnls_1")
#a = np.array(a)
#a = torch.tensor(a, dtype=torch.float32)
#print(a.shape)
#print(a[0].shape)
#print(a.squeeze(0).shape)
#show_plt(a[0], "Flair")
#b = a.unsqueeze(0)
#print(b.shape)


# In[ ]:


#def get_last_conv_name(model):
#    layer_name = None
#    for name, m in model.named_modules():
#        if isinstance(m, nn.Conv3d):
#            layer_name = name
#            #print(layer_name)
#    return layer_name

#def get_last_conv_name_(model):
def get_last_conv_name_(model):
    layers_name = []
    layer_name = None
    mm = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            layer_name = name
            mm = m
            #print(layer_name)
            #print(m)
            layers_name.append(layer_name)
    return layers_name[-1]









# In[ ]:


#model = Model()
#model.to(device)
#checkpoint = torch.load(modelfile)
#model.load_state_dict(checkpoint["model_state_dict"])
#print(get_last_conv_name(model))

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#modelfile_train_loss = "/data/data_cotune_l2/T13D-densenet_min_train_loss_model_classification3d_array-12-56.pth"
#modelfile_valid_loss = "/data/data_cotune_l2/T13D-densenet_min_valid_loss_model_classification3d_array-12-56.pth"
#modelfile_valid_ROC = "/data/data_cotune_l2/T13D-densenet_max_valid_ROC_model_classification3d_array-12-56.pth"
#modelfile_valid_fscore = "/data/data_cotune_l2/T13D-densenet_max_valid_fscore_model_classification3d_array-12-56.pth" # train_all_type(df_train, df_valid, mri_type="T13D-T2")#"all"
#checkpoint_new_train_loss = torch.load(modelfile_train_loss)
#checkpoint_new_valid_loss = torch.load(modelfile_valid_loss)
#checkpoint_new_valid_ROC = torch.load(modelfile_valid_ROC)
#checkpoint_new_valid_fscore = torch.load(modelfile_valid_fscore)
#model.load_state_dict(checkpoint_new["model_state_dict"], strict=False)
# class Net_New_train_loss(nn.Module):
#     def __init__(self):
#         super(Net_New_train_loss, self).__init__()
#         model = ResidualAttentionModel_56()
#         model.load_state_dict(checkpoint_new_train_loss["model_state_dict"])
#
#     def forward(self, x):
#         out = model(x) # target
#         return out
#
# class Net_New_valid_loss(nn.Module):
#     def __init__(self):
#         super(Net_New_valid_loss, self).__init__()
#         model = ResidualAttentionModel_56()
#         model.load_state_dict(checkpoint_new_valid_loss["model_state_dict"])
#
#     def forward(self, x):
#         out = model(x)  # target
#         return out
#
# class Net_New_valid_ROC(nn.Module):
#     def __init__(self):
#         super(Net_New_valid_ROC, self).__init__()
#         model = ResidualAttentionModel_56()
#         model.load_state_dict(checkpoint_new_valid_ROC["model_state_dict"])
#
#     def forward(self, x):
#         out = model(x)  # target
#         return out
#
# class Net_New_valid_fscore(nn.Module):
#     def __init__(self):
#         super(Net_New_valid_fscore, self).__init__()
#         model = ResidualAttentionModel_56()
#         model.load_state_dict(checkpoint_new_valid_fscore["model_state_dict"])
#
#     def forward(self, x):
#         out = model(x)  # target
#         return out

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_train_loss = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)  # .to(device)
# modelfile_train_loss = "/data/data_T13D/['T13D']-resnet_min_train_loss_model_classification3d_array-12.pth"
# checkpoint_new_train_loss = torch.load(modelfile_train_loss)
# model_train_loss.load_state_dict(checkpoint_new_train_loss["model_state_dict"])
# model_train_loss.to(device)


# model_valid_loss = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)
# modelfile_valid_loss = "/data/data_T13D/['T13D']-resnet_min_valid_loss_model_classification3d_array-12.pth"
# checkpoint_new_valid_loss = torch.load(modelfile_valid_loss)
# model_valid_loss.load_state_dict(checkpoint_new_valid_loss["model_state_dict"])
# model_valid_loss.to(device)
#
#
# model_valid_ROC = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)
# modelfile_valid_ROC = "/data/data_T13D/['T13D']-resnet_max_valid_ROC_model_classification3d_array-12.pth"
# checkpoint_new_valid_ROC = torch.load(modelfile_valid_ROC)
# model_valid_ROC.load_state_dict(checkpoint_new_valid_ROC["model_state_dict"])
# model_valid_ROC.to(device)
#
#
model_valid_fscore = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)
modelfile_valid_fscore = "/data/data_Flair/['Flair']-resnet_max_valid_fscore_model_classification3d_array-12.pth"  # train_all_type(df_train, df_valid, mri_type="T13D-T2")#"all"
checkpoint_new_valid_fscore = torch.load(modelfile_valid_fscore)
model_valid_fscore.load_state_dict(checkpoint_new_valid_fscore["model_state_dict"])
model_valid_fscore.to(device)

# get all the model children as list
#model_children = list(model.children())
#for i in model_children:
#    print(i)
#    print("\n")
#print(len(model_children))

def loader_new(random_state):
    # df_train, df_valid, dummies_train, products_train, y_train, dummies_valid, products_valid, y_valid = ran_state(
    #     random_state)
    valid_transforms = Compose(
        [
            # AsChannelFirst(),
            Resize(spatial_size=(128, 128, 128)),
            # RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            # RandFlip(spatial_axis=0, prob=0.5),
            ScaleIntensity(),
            EnsureType(),
        ]
    )
    data_directory = "/data/down"
    valid_df = pd.read_csv("{}/Overall_Survival_1year.csv".format(data_directory))
    #valid_df = pd.read_csv("{}/test_new.csv".format(data_directory))
    # display(train_df)
    X = valid_df["Patient"].values
    y = valid_df["OS 1 year"]
    dummies_valid = pd.get_dummies(y)  # Classification
    products_valid = dummies_valid.columns
    y_valid_dummies = dummies_valid.values

    valid_data_retriever = Dataset(
        paths=X,
        targets=y_valid_dummies,
        norm_set_of_files=norm_set_of_files,
        split="Pcnls_baseline",
        transforms=valid_transforms
    )
    valid_loader = DataLoader(  # torch_data.DataLoader(
        valid_data_retriever,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate
    )

    # valid_data_retriever_new = Dataset_new(
    #     paths=X,
    #     targets=y_valid_dummies,
    #     norm_set_of_files_new=norm_set_of_files_new,
    #     split="Pcnls_baseline",
    #     transforms=valid_transforms
    #     )
    # valid_loader_new = DataLoader(  # torch_data.DataLoader(
    #     valid_data_retriever_new,
    #     batch_size=16,
    #     shuffle=False,
    #     num_workers=2,
    #     pin_memory=torch.cuda.is_available(),
    #     collate_fn=pad_list_data_collate
    # )
    # return [valid_loader],  [valid_loader_new], products_valid,  y_valid_dummies
    return [valid_loader], products_valid, y_valid_dummies

random_state = 12
# valid_loader_list,  valid_loader_new_list, products_valid,  y_valid_dummies = loader_new(random_state)
# test_loader_list = valid_loader_list
# test_loader_new_list = valid_loader_new_list
valid_loader_list,  products_valid,  y_valid_dummies = loader_new(random_state)
test_loader_list = valid_loader_list
products_test = products_valid
y_test_dummies = y_valid_dummies
forward_passes = 1
n_classes = 2


l1 = [model_valid_fscore]#, model_valid_loss, model_valid_ROC, model_valid_fscore]
l2 = ["valid_fscore"]#, "valid_loss", "valid_ROC", "valid_fscore"]
labels = [0, 1]
list = []
list_labels = []
seg = []
patients = []
for test_loader in test_loader_list:
    for step, batch in enumerate(test_loader):
        list.extend(batch["X"])
        list_labels.extend(batch["y"].argmax(dim=1))
        patients.extend(batch["id"])
print("Number of Images: {}".format(len(list)))

# for test_loader_new in test_loader_new_list:
#     for step, batch in enumerate(test_loader_new):
#         seg.extend(batch["X"])
# print("Number of Segs: {}".format(len(seg)))




def medCAM(img, id, modelfile, model, mri_types, index, truelabel
           # = ("Flair", "T1", "T13D", "T2")
           ):
    img = img.unsqueeze(0).to(device)
    for i, j in enumerate(mri_types):
        # for i in range(len(list)):
        # layer_name = medcam.get_layers(model)[-6]
        layer_name = get_last_conv_name_(model)
        # print(layer_name)
        model.eval()
        prelabel = model(img)
        prelabel = torch.sigmoid(prelabel).argmax(dim=1).item()
        #prelabel = torch.Tensor.cpu(prelabel)
        #print(prelabel)
        Model = medcam.inject(model, output_dir="/data/data_Flair/resnet_attention_maps", backend="gcam",
                              layer=layer_name, label=index, replace=True,
                              return_score=True)
        Model.eval()
        output = Model(img)
        # print(len(output))
        xyz = medcam.save(output[0].squeeze(0).squeeze(0),
                          "/data/data_Flair/resnet_attention_maps/modelfile_{}/patients_{}/type_{}-layer_{}-index_{}-truelabel_{}-prelabel_{}".format(modelfile, id, j, layer_name, index, truelabel, prelabel), heatmap=True,
                          raw_input=None)
        x = torch.Tensor.cpu(output[0].squeeze(0).squeeze(0))
        y = torch.Tensor.cpu(img[0][i])
        z = (x * 0.3 + y * 0.5) / 225.

        xyz_ = medcam.save(z, "/data/data_Flair/resnet_attention_maps_/modelfile_{}/patients_{}/type_{}-layer_{}-index_{}-truelabel_{}-prelabel_{}".format(modelfile, id, j, layer_name, index, truelabel, prelabel), heatmap=True,
                          raw_input=None)


        path = "/data/data_Flair/resnet_feature_maps/add-modelfile_{}/patients_{}".format(modelfile, id)
        path_old = "/data/data_Flair/old_resnet_feature_maps/add-modelfile_{}/patients_{}".format(modelfile, id)
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)

        folder_old = os.path.exists(path_old)
        if not folder_old:
            os.makedirs(path_old)

        show_plt_old(y, "{}/type_{}-layer_{}-index_{}-truelabel_{}-prelabel_{}.jpg".format(path_old, j, layer_name, index, truelabel, prelabel))
        # show_plt_(x, "/data/data_Flair/resnet_feature_maps/raw-modelfile_{}/patients_{}/type_{}-layer_{}-index_{}-truelabel_{}-prelabel_{}".format(modelfile, id, j, layer_name, index, truelabel, prelabel))
        show_plt_(z, "{}/type_{}-layer_{}-index_{}-truelabel_{}-prelabel_{}.jpg".format(path, j, layer_name, index, truelabel, prelabel))


for model, modelfile in zip(l1, l2):
    for index in labels:
        for i in range(len(list)):
            medCAM(img=list[i], id=patients[i], modelfile=modelfile, model=model, mri_types=['Flair'], index=index, truelabel=list_labels[i])





