#!/usr/bin/python
# coding: utf-8

# In[1]:





# In[2]:



# In[3]:



# In[ ]:

import train_resnet
from train_resnet import *


# In[ ]:




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

def get_last_conv_name_(model):
    layer_name = None
    mm = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            layer_name = name
            mm = m
            #print(m)
            #print(layer_name)
    return layer_name

#modelfile = "all-cofinetuneresnet_min_loss_model_classification3d_array.pth"
#checkpoint_new = torch.load(modelfile)
#class Net_New(nn.Module):
#    def __init__(self):
#        super(Net_New, self).__init__()
#        self.net = Net()
#        self.net.load_state_dict(checkpoint_new["model_state_dict"])
         

#    def forward(self, x):
#        out_1, out_2 = self.net(x)

#        return out_2

#print(get_last_conv_name_(Net_New()))

import torch
#torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# In[ ]:







# In[ ]:


#def medCAM(channelimg, modelfile, mri_types
#           #= ("Flair", "T1", "T13D", "T2")
#          ):
#    for i, j in enumerate(mri_types):
#        #print(j)
#        channelimg = np.array(channelimg)
#        tensor = torch.tensor(channelimg, dtype=torch.float32).unsqueeze(0)
#
#        tensor = tensor.cuda()
#
#        torch.cuda.empty_cache()
#        #model = Model()
#       # model = monai.networks.nets.resnet.resnet50(n_input_channels=2, n_classes=2)
#        #model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=3)#.to(device)
#       # model.to(device)
#        Model = Net_New()#.cuda()
#        Model.to(device)
#        #layer_name = get_last_conv_name(model)
#        #print(layer_name)
#        name = get_last_conv_name_(Model)
#        Model = medcam.inject(Model, output_dir="/data/data/resnet_attention_maps", backend="gcam", layer=name, label=1, replace=True,
#                              return_score=True)
#        Model.eval()
#        output = Model(tensor)
#        #print(len(output))
#        #print(output[1])
#        print(output[0].shape)
#        xyz = medcam.save(output[0].squeeze(0).squeeze(0), "/data/data/resnet_attention_maps/{}/{}".format(j, modelfile), heatmap=True, raw_input=None)
#        x = torch.Tensor.cpu(output[0].squeeze(0).squeeze(0))
#        y = channelimg[i]
#        z = x + y
#
#        xyz_ = medcam.save(z, "/data/data/resnet_attention_maps_/{}/{}".format(j, modelfile), heatmap=True, raw_input=None)
#
#        #show_plt(z)
#        show_plt_(z, j)
#        #show_plt_(torch.Tensor.cpu(output[0].squeeze(0).squeeze(0)))

#print("done")








#######################################################################################
# set_of_files = ["SegT1.seg.nii.gz", "SegT2.seg.nii.gz"]
# norm_set_of_files = ["SegT1.seg/SegT1.seg_trans.nii.gz", "SegT2.seg/SegT2.seg_trans.nii.gz"]
# data_directory="/data/down"
# df_test = pd.read_csv("{}/test.csv".format(data_directory))
# scan_ids_test = list(df_test["Patient"])
# for scan_id_test in scan_ids_test:
#     get_foreground_from_set_of_files(scan_id_test, set_of_files, background_value=0, tolerance=0.00001)
#
# dummies_test = pd.get_dummies(df_test["OS 1 year"]) # Classification
# cats = [0,1]
# dummies_test = dummies_test.T.reindex(cats).T.fillna(0)
# products_test = dummies_test.columns
# y_test = dummies_test.values
# #print(products_test)
# #print(y_test)

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import csv
import numpy as np
import pandas as pd
# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, save_path, title="Confusion Matrix", cmap=plt.cm.Blues):
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Calibri"
    plt.figure()
#    plt.imshow(cm, interpolation="nearest", cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(names))
#    plt.xticks(tick_marks, names)#, rotation=45)
#    plt.yticks(tick_marks, names)

#    for first_index in range(len(names)):
#        for second_index in range(len(names)):  # [first_index])):
#            temp = cm[first_index][second_index]
#            if temp == 0.0 or temp == 100.0:
#                plt.text(first_index, second_index, int(temp), va='center',
#                         ha='center')
#                # fontsize=13.5)
#            else:
#                plt.text(first_index, second_index, r'{0:.2f}'.format(temp), va='center',
#                         ha='center')
#                # fontsize=13.5)

#    plt.tight_layout()
#    plt.gcf().subplots_adjust(bottom=0.15)
#    plt.ylabel("True Label")
#    plt.xlabel("Predicted Label")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(
        include_values=True,
        cmap="viridis",
        ax=None,
        xticks_rotation="horizontal",
        #values_format="d"
        )
    disp.ax_.set_title(title)
    plt.savefig("/data/data_T13D/{}_cm.jpg".format(save_path))
    plt.show()

# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred, y, save_path):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    # print(roc_auc)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Calibri"
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (Area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="best")
    plt.savefig("/data/data_T13D/{}_roc.jpg".format(save_path))
    plt.show()




def get_monte_carlo_predictions(loader_list,
                                forward_passes,
                                model,
                                n_classes,
                                y_names,
                                y_,
                                save_path):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
    n_samples = len(y_)
    # print(n_samples)
   
    dropout_predictions = np.empty((0, len(loader_list), n_samples, n_classes))
    internal_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    # for_results = []
    # model.eval()
    for i in range(forward_passes):
        for loader in loader_list:
            predictions = np.empty((0, n_classes))
            model.eval()
            num_correct = 0.0
            metric_count = 0
            with torch.no_grad():
                for step, batch in enumerate(loader, 1):
                    inputs = batch["X"].to(device)
                    labels = batch["y"].to(device)
                    outputs = model(inputs)
                    outputs = torch.sigmoid(outputs)  # shape (n_samples, n_classes)
                    predictions = np.vstack((predictions, outputs.cpu().numpy()))
            internal_predictions = np.vstack((internal_predictions,
                                             predictions[np.newaxis, :, :]))
        #internal_predictions - shape (len(loader_list), n_samples, n_classes)
        #dropout_predictions - shape (forward_passes, len(loader_list), n_samples, n_classes)
        dropout_predictions = np.vstack((dropout_predictions,
                                         internal_predictions[np.newaxis, :, :, :]))
        
        

    # for_results += [(ps)]
    # print(for_results)
    # ps = np.mean([r[0] for r in for_results], 0)
    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)
    mean = np.mean(mean, axis=0)
    # shape (n_samples, n_classes)
    mean_1D = np.argmax(mean, axis=1)
    y_1D = np.argmax(y_, axis=1)
    print(y_1D)
    tf = mean_1D == y_1D
    num_correct = np.sum(tf != 0)
    metric_count = len(mean_1D)
    auc = num_correct / metric_count
    precision, recall, fscore, support = precision_recall_fscore_support(y_1D, mean_1D, average="macro")
    cm = confusion_matrix(y_1D, mean_1D)

    # np.set_printoptions(precision=2)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)

    plot_confusion_matrix(cm_normalized, y_names, save_path, title="Confusion Matrix of T1Gd-B", cmap=plt.cm.Blues)

    # plot_confusion_matrix(cm, y_names, title='Normalized confusion matrix')
    # plt.savefig("/data/data_T13D/cm.jpg")
    mean_new = pd.DataFrame(mean)
    mean_new = mean_new.iloc[:, 1]
    mean_new = list(mean_new)
    #print(mean_new)
    plot_roc(mean_new, y_1D, save_path)
    ROC = roc_auc_score(y_1D, mean_new)

    print("ROC =", ROC)
    print("auc =", auc)
    print("precision =", precision)
    print("recall =", recall)
    print("fscore =", fscore)

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = - np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(np.sum(- dropout_predictions * np.log(dropout_predictions + epsilon),
                                           axis=-1), axis=0)  # shape (n_samples,)

    return mean, variance, entropy, mutual_info, ROC, auc, precision, recall, fscore  # , support


# df_test = df_valid
# y_test = y_valid
# products_test = products_valid
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
    valid_df = pd.read_csv("{}/test_new.csv".format(data_directory))
    #valid_df = pd.read_csv("{}/Overall_Survival_1year_new.csv".format(data_directory))
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
    return [valid_loader],  products_valid,  y_valid_dummies

random_state = 12
valid_loader_list,  products_valid,  y_valid_dummies = loader_new(random_state)
test_loader_list = valid_loader_list
products_test = products_valid
y_test_dummies = y_valid_dummies
forward_passes = 1
n_classes = 2


def predict(model, random_state, forward_passes, n_classes, save_path):
    # test_transforms = Compose(
    #     [
    #         # AsChannelFirst(),
    #         Resize(spatial_size=(128,128,128)),
    #         # RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
    #         # RandFlip(spatial_axis=0, prob=0.5),
    #         ScaleIntensity(),
    #         EnsureType(),
    #     ]
    # )
    #
    # test_data_retriever = Dataset(
    #     paths=df_test["Patient"].values,
    #     targets=y_test,
    #     norm_set_of_files=norm_set_of_files,
    #     split="Pcnls_baseline",
    #     transforms=test_transforms
    # )
    #
    #
    # test_loader = DataLoader(  # torch_data.DataLoader(
    #     test_data_retriever,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=torch.cuda.is_available(),
    #     collate_fn=pad_list_data_collate
    # )
    #

    #model = Net_New()
    #model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=3)#.to(device)
    #model.to(device)

    mcp_test = get_monte_carlo_predictions(loader_list=test_loader_list,
                                           forward_passes=forward_passes,
                                           model=model,
                                           n_classes=n_classes,
                                           y_names=products_test,
                                           y_=y_test_dummies,
                                           save_path=save_path)
    print(mcp_test)
    print("\n")
    return mcp_test





#
# model_train_loss = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)  # .to(device)
# modelfile_train_loss = "/data/data_T13D/['T13D']-resnet_min_train_loss_model_classification3d_array-12.pth"
# checkpoint_new_train_loss = torch.load(modelfile_train_loss)
# model_train_loss.load_state_dict(checkpoint_new_train_loss["model_state_dict"])
# model_train_loss.to(device)
# pre_train_loss = predict(model_train_loss, random_state, forward_passes, n_classes, "train_loss")

# model_valid_loss = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)
# modelfile_valid_loss = "/data/data_T13D/['T13D']-resnet_min_valid_loss_model_classification3d_array-12.pth"
# checkpoint_new_valid_loss = torch.load(modelfile_valid_loss)
# model_valid_loss.load_state_dict(checkpoint_new_valid_loss["model_state_dict"])
# model_valid_loss.to(device)
# pre_valid_loss = predict(model_valid_loss, random_state, forward_passes, n_classes, "valid_loss")
#
# model_valid_ROC = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)
# modelfile_valid_ROC = "/data/data_T13D/['T13D']-resnet_max_valid_ROC_model_classification3d_array-12.pth"
# checkpoint_new_valid_ROC = torch.load(modelfile_valid_ROC)
# model_valid_ROC.load_state_dict(checkpoint_new_valid_ROC["model_state_dict"])
# model_valid_ROC.to(device)
# pre_valid_ROC = predict(model_valid_ROC, random_state, forward_passes, n_classes, "valid_ROC")
#
model_valid_fscore = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)
modelfile_valid_fscore = "/data/data_T13D/['T13D']-resnet_max_valid_fscore_model_classification3d_array-12.pth"  # train_all_type(df_train, df_valid, mri_type="T13D-T2")#"all"
checkpoint_new_valid_fscore = torch.load(modelfile_valid_fscore)
model_valid_fscore.load_state_dict(checkpoint_new_valid_fscore["model_state_dict"])
model_valid_fscore.to(device)
pre_valid_fscore = predict(model_valid_fscore, random_state, forward_passes, n_classes, "valid_fscore")
