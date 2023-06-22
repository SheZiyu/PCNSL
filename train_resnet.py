
#!/usr/bin/python
# coding: utf-8


from dataset_resnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(50)
model = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)#.to(device)
model.to(device)
random_state = 12
#model_file = "/data/MedicalNet_pytorch_files2/resnet_50.pth"
#checkpoint = torch.load(model_file)
#model.load_state_dict(checkpoint["state_dict"], strict=False)
#print(get_last_conv_name(model_resnet50))
 



#class Model(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.net = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=2)
#        n_features = self.net._fc.in_features
#        self.net._fc = nn.Linear(in_features=n_features, out_features=2, bias=True)
#    
#    def forward(self, x):
#        out = self.net(x)
#        return out
 

 


class Trainer:
    def __init__(
            self,
            model,
            device,
            criterion,
            optimizer,
            scheduler,
            random_state
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.random_state = random_state

        self.best_train_loss = np.inf
        self.best_valid_loss = np.inf
        self.best_valid_ROC = -np.inf
        self.best_valid_fscore = -np.inf
        self.n_patience_train_loss = 0
        self.n_patience_valid_loss = 0
        self.n_patience_valid_ROC = 0
        self.n_patience_valid_fscore = 0
        self.lastmodel_train_loss = None
        self.lastmodel_valid_loss = None
        self.lastmodel_valid_ROC = None
        self.lastmodel_valid_fscore = None
        self.best_train_loss_list = []
        self.best_valid_loss_list = []
        self.best_valid_ROC_list = []
        self.best_valid_fscore_list = []

    def fit_train_loss(self, epochs, train_loader_list, valid_loader_list, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_Flair/resnet_train_loss")
        start_time = time.time()
        self.valid_auc_new_list = []
        self.valid_ROC_new_list = []
        self.valid_precision_new_list = []
        self.valid_recall_new_list = []
        self.valid_fscore_new_list = []

        for i, (train_loader, valid_loader) in enumerate(zip(train_loader_list, valid_loader_list)):
            print("round: {}".format(i+1))
            print("\n")
            self.n_patience_train_loss = 0
            self.n_patience_valid_loss = 0
            self.n_patience_valid_ROC = 0
            self.n_patience_valid_fscore = 0
            torch.manual_seed(50)
            self.model = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)  # .to(device)
            self.model.to(device)
            random_state = 12
            for n_epoch in range(1, epochs + 1):
                self.info_message("EPOCH: {}", n_epoch)

                train_loss, train_time = self.train_epoch(train_loader)

                valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(
                    valid_loader)

                self.info_message(
                    "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                    n_epoch, train_loss, train_time
                )
                writer.add_scalar("Train_loss", train_loss, n_epoch)


                self.info_message(
                    "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                    n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
                )
                writer.add_scalar("Valid_loss", valid_loss, n_epoch)
                writer.add_scalar("Valid_auc", valid_auc, n_epoch)
                writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)
                writer.add_scalar("Valid_precision", precision, n_epoch)
                writer.add_scalar("Valid_recall", recall, n_epoch)
                writer.add_scalar("Valid_fscore", fscore, n_epoch)

                if self.best_train_loss > train_loss:
                    self.save_model_train_loss(n_epoch, save_path)
                    self.info_message(
                        "Loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                        self.best_train_loss, train_loss, self.lastmodel_train_loss
                    )
                    self.best_train_loss = train_loss
                    self.valid_auc_new = valid_auc
                    self.valid_ROC_new = valid_ROC
                    self.valid_precision_new = precision
                    self.valid_recall_new = recall
                    self.valid_fscore_new = fscore
                    self.n_patience_train_loss = 0
                else:
                    self.n_patience_train_loss += 1

                if self.n_patience_train_loss >= patience or n_epoch == epochs:
                    print("self.n_patience_train_loss: {}".format(self.n_patience_train_loss))
                    print("n_epoch: {}".format(n_epoch))
                    print("append......................................................................................................")
                    self.best_train_loss_list.append(self.best_train_loss)
                    self.valid_auc_new_list.append(self.valid_auc_new)
                    self.valid_ROC_new_list.append(self.valid_ROC_new)
                    self.valid_precision_new_list.append(self.valid_precision_new)
                    self.valid_recall_new_list.append(self.valid_recall_new)
                    self.valid_fscore_new_list.append(self.valid_fscore_new)

                    if self.n_patience_train_loss >= patience:
                        self.info_message("\nTrain loss didn't improve last {} epochs.", patience)
                    break
        time_took = time.time() - start_time
        print("train completed, best_train_loss: {}, time took: {}.".format(np.mean(self.best_train_loss_list), hms_string(time_took)))
        writer.close()
        print(self.best_train_loss_list)
        print("train completed, auc_list: {}, mean_auc: {}.".format(self.valid_auc_new_list,np.mean(self.valid_auc_new_list)))
        print("train completed, ROC_list: {}, mean_ROC: {}.".format(self.valid_ROC_new_list,np.mean(self.valid_ROC_new_list)))
        print("train completed, precision_list: {}, mean_precison: {}.".format(self.valid_precision_new_list,np.mean(self.valid_precision_new_list)))
        print("train completed, recall_list: {}, mean_recall: {}.".format(self.valid_recall_new_list,np.mean(self.valid_recall_new_list)))
        print("train completed, fscore_list: {}, mean_fscore: {}.".format(self.valid_fscore_new_list,np.mean(self.valid_fscore_new_list)))
        return np.mean(self.best_train_loss_list)

    def fit_valid_loss(self, epochs, train_loader_list, valid_loader_list, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_Flair/resnet_valid_loss")
        start_time = time.time()
        self.valid_auc_new_list = []
        self.valid_ROC_new_list = []
        self.valid_precision_new_list = []
        self.valid_recall_new_list = []
        self.valid_fscore_new_list = []
        for i, (train_loader, valid_loader) in enumerate(zip(train_loader_list, valid_loader_list)):
            print("round: {}".format(i + 1))
            print("\n")
            self.n_patience_train_loss = 0
            self.n_patience_valid_loss = 0
            self.n_patience_valid_ROC = 0
            self.n_patience_valid_fscore = 0
            torch.manual_seed(50)
            self.model = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)  # .to(device)
            self.model.to(device)
            random_state = 12
            for n_epoch in range(1, epochs + 1):
                self.info_message("EPOCH: {}", n_epoch)

                train_loss, train_time = self.train_epoch(train_loader)

                valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(
                    valid_loader)

                self.info_message(
                    "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                    n_epoch, train_loss, train_time
                )
                writer.add_scalar("Train_loss", train_loss, n_epoch)


                self.info_message(
                    "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                    n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
                )
                writer.add_scalar("Valid_loss", valid_loss, n_epoch)
                writer.add_scalar("Valid_auc", valid_auc, n_epoch)
                writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)
                writer.add_scalar("Valid_precision", precision, n_epoch)
                writer.add_scalar("Valid_recall", recall, n_epoch)
                writer.add_scalar("Valid_fscore", fscore, n_epoch)

                if self.best_valid_loss > valid_loss:
                    self.save_model_valid_loss(n_epoch, save_path)
                    self.info_message(
                        "Loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                        self.best_valid_loss, valid_loss, self.lastmodel_valid_loss
                    )
                    self.best_valid_loss = valid_loss
                    self.valid_auc_new = valid_auc
                    self.valid_ROC_new = valid_ROC
                    self.valid_precision_new = precision
                    self.valid_recall_new = recall
                    self.valid_fscore_new = fscore
                    self.n_patience_valid_loss = 0
                else:
                    self.n_patience_valid_loss += 1

            # if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            # else:
            #    self.n_patience += 1

                if self.n_patience_valid_loss >= patience or n_epoch == epochs:
                    print("self.n_patience_valid_loss: {}".format(self.n_patience_valid_loss))
                    print("n_epoch: {}".format(n_epoch))
                    print("append......................................................................................................")
                    self.best_valid_loss_list.append(self.best_valid_loss)
                    self.valid_auc_new_list.append(self.valid_auc_new)
                    self.valid_ROC_new_list.append(self.valid_ROC_new)
                    self.valid_precision_new_list.append(self.valid_precision_new)
                    self.valid_recall_new_list.append(self.valid_recall_new)
                    self.valid_fscore_new_list.append(self.valid_fscore_new)
                    if self.n_patience_valid_loss >= patience:
                        self.info_message("\nValid loss didn't improve last {} epochs.", patience)
                    break

        time_took = time.time() - start_time
        print("train completed, best_valid_loss: {}, time took: {}.".format(np.mean(self.best_valid_loss_list), hms_string(time_took)))
        writer.close()
        print(self.best_valid_loss_list)
        print("train completed, auc_list: {}, mean_auc: {}.".format(self.valid_auc_new_list,
                                                                    np.mean(self.valid_auc_new_list)))
        print("train completed, ROC_list: {}, mean_ROC: {}.".format(self.valid_ROC_new_list,
                                                                    np.mean(self.valid_ROC_new_list)))
        print("train completed, precision_list: {}, mean_precison: {}.".format(self.valid_precision_new_list,
                                                                               np.mean(self.valid_precision_new_list)))
        print("train completed, recall_list: {}, mean_recall: {}.".format(self.valid_recall_new_list,
                                                                          np.mean(self.valid_recall_new_list)))
        print("train completed, fscore_list: {}, mean_fscore: {}.".format(self.valid_fscore_new_list,
                                                                          np.mean(self.valid_fscore_new_list)))
        return np.mean(self.best_valid_loss_list)
    
    def fit_valid_ROC(self, epochs, train_loader_list, valid_loader_list, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_Flair/resnet_valid_ROC")
        start_time = time.time()
        self.valid_auc_new_list = []
        self.valid_ROC_new_list = []
        self.valid_precision_new_list = []
        self.valid_recall_new_list = []
        self.valid_fscore_new_list = []
        for i, (train_loader, valid_loader) in enumerate(zip(train_loader_list, valid_loader_list)):
            print("round: {}".format(i + 1))
            print("\n")
            self.n_patience_train_loss = 0
            self.n_patience_valid_loss = 0
            self.n_patience_valid_ROC = 0
            self.n_patience_valid_fscore = 0
            torch.manual_seed(50)
            self.model = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)  # .to(device)
            self.model.to(device)
            random_state = 12
            for n_epoch in range(1, epochs + 1):
                self.info_message("EPOCH: {}", n_epoch)

                train_loss, train_time = self.train_epoch(train_loader)

                valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(
                    valid_loader)

                self.info_message(
                    "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                    n_epoch, train_loss, train_time
                )
                writer.add_scalar("Train_loss", train_loss, n_epoch)


                self.info_message(
                    "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                    n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
                )
                writer.add_scalar("Valid_loss", valid_loss, n_epoch)
                writer.add_scalar("Valid_auc", valid_auc, n_epoch)
                writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)
                writer.add_scalar("Valid_precision", precision, n_epoch)
                writer.add_scalar("Valid_recall", recall, n_epoch)
                writer.add_scalar("Valid_fscore", fscore, n_epoch)

                if self.best_valid_ROC < valid_ROC:
                    self.save_model_valid_ROC(n_epoch, save_path)
                    self.info_message(
                        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
                        self.best_valid_ROC, valid_ROC, self.lastmodel_valid_ROC
                    )
                    self.best_valid_ROC = valid_ROC
                    self.valid_auc_new = valid_auc
                    self.valid_ROC_new = valid_ROC
                    self.valid_precision_new = precision
                    self.valid_recall_new = recall
                    self.valid_fscore_new = fscore
                    self.n_patience_valid_ROC = 0
                else:
                    self.n_patience_valid_ROC += 1

            # if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            # else:
            #    self.n_patience += 1

                if self.n_patience_valid_ROC >= patience or n_epoch == epochs:
                    print("self.n_patience_valid_ROC: {}".format(self.n_patience_valid_ROC))
                    print("n_epoch: {}".format(n_epoch))
                    print("append......................................................................................................")
                    self.best_valid_ROC_list.append(self.best_valid_ROC)
                    self.valid_auc_new_list.append(self.valid_auc_new)
                    self.valid_ROC_new_list.append(self.valid_ROC_new)
                    self.valid_precision_new_list.append(self.valid_precision_new)
                    self.valid_recall_new_list.append(self.valid_recall_new)
                    self.valid_fscore_new_list.append(self.valid_fscore_new)
                    if self.n_patience_valid_ROC >= patience:
                        self.info_message("\nValid ROC didn't improve last {} epochs.", patience)
                    break
        time_took = time.time() - start_time
        print("train completed, best_valid_ROC: {}, time took: {}.".format(np.mean(self.best_valid_ROC_list), hms_string(time_took)))
        writer.close()
        print(self.best_valid_ROC_list)
        print("train completed, auc_list: {}, mean_auc: {}.".format(self.valid_auc_new_list,
                                                                    np.mean(self.valid_auc_new_list)))
        print("train completed, ROC_list: {}, mean_ROC: {}.".format(self.valid_ROC_new_list,
                                                                    np.mean(self.valid_ROC_new_list)))
        print("train completed, precision_list: {}, mean_precison: {}.".format(self.valid_precision_new_list,
                                                                               np.mean(self.valid_precision_new_list)))
        print("train completed, recall_list: {}, mean_recall: {}.".format(self.valid_recall_new_list,
                                                                          np.mean(self.valid_recall_new_list)))
        print("train completed, fscore_list: {}, mean_fscore: {}.".format(self.valid_fscore_new_list,
                                                                          np.mean(self.valid_fscore_new_list)))
        return np.mean(self.best_valid_ROC_list)

    def fit_valid_fscore(self, epochs, train_loader_list, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_Flair/resnet_valid_fscore")
        start_time = time.time()
        self.valid_auc_new_list = []
        self.valid_ROC_new_list = []
        self.valid_precision_new_list = []
        self.valid_recall_new_list = []
        self.valid_fscore_new_list = []

        for i, train_loader in enumerate(train_loader_list):
            print("round: {}".format(i + 1))
            print("\n")
            self.n_patience_train_loss = 0
            self.n_patience_valid_loss = 0
            self.n_patience_valid_ROC = 0
            self.n_patience_valid_fscore = 0
            torch.manual_seed(50)
            self.model = monai.networks.nets.resnet.resnet50(n_input_channels=1, n_classes=2)  # .to(device)
            self.model.to(device)
            random_state = 12
            for n_epoch in range(1, epochs + 1):
                self.info_message("EPOCH: {}", n_epoch)

                train_loss, train_time = self.train_epoch(train_loader)

                self.info_message(
                    "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                    n_epoch, train_loss, train_time
                )
                writer.add_scalar("Train_loss", train_loss, n_epoch)






            # if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            # else:
            #    self.n_patience += 1
                if n_epoch == epochs:
                    self.save_model_valid_fscore(n_epoch, save_path)
        time_took = time.time() - start_time
        print("train completed,  time took: {}.".format(hms_string(time_took)))
        writer.close()

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            train_inputs = batch["X"].to(self.device)
            train_labels = batch["y"].to(self.device)
            ind = train_labels.argmax(dim=1)
            ind = torch.Tensor.cpu(ind)

            # weight_decay, dropout
            train_outputs = self.model(train_inputs)
            loss = - train_labels * self.criterion(train_outputs)
            loss = torch.mean(torch.sum(loss, dim=-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss += loss.item()

            message = "Train Step {}/{}, train_loss: {:.4f}"
            self.info_message(message, step, len(train_loader), sum_loss / step, end="\r")
        self.scheduler.step()
        return sum_loss / len(train_loader), int(time.time() - t)

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []
        pred_all = []
        num_correct = 0.0
        metric_count = 0
        with torch.no_grad():
            for step, batch in enumerate(valid_loader, 1):
                val_inputs = batch["X"].to(self.device)
                val_labels = batch["y"].to(self.device)
                ind = val_labels.argmax(dim=1)
                ind = torch.Tensor.cpu(ind)

                val_outputs = self.model(val_inputs)
                loss = - val_labels * self.criterion(val_outputs)
                loss = torch.mean(torch.sum(loss, dim=-1))

                sum_loss += loss.item()

                value = torch.eq(torch.sigmoid(val_outputs).argmax(dim=1), val_labels.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()

                y_all.extend(batch["y"].argmax(dim=1).tolist())
                outputs_all.extend(torch.sigmoid(val_outputs).argmax(dim=1).tolist())
                pred_all.extend(torch.sigmoid(val_outputs).tolist())

                message = "Valid Step {}/{}, valid_loss: {:.4f}"
                self.info_message(message, step, len(valid_loader), sum_loss / step, end="\r")

            auc = num_correct / metric_count
            pred_all_new = pd.DataFrame(pred_all)
            print(pred_all_new)
            print(y_all)
            pred_all_new = pred_all_new.iloc[:, 1]
            pred_all_new = list(pred_all_new)
            # print(mean_new)
            ROC = roc_auc_score(y_all, pred_all_new)
            # metric_values.append(metric)
            # auc = roc_auc_score(y_all, torch.Tensor.cpu(outputs))
            # outputs_all = [1 if y > 0.5 else 0 for y in outputs_all]
            # auc_ = roc_auc_score(y_all, outputs_all)
            precision, recall, fscore, support = precision_recall_fscore_support(y_all, outputs_all, average='macro')

            return sum_loss / len(valid_loader), auc, ROC, int(time.time() - t), precision, recall, fscore, support

    def save_model_train_loss(self, n_epoch, save_path):
        self.lastmodel_train_loss = "/data/data_Flair/{}-resnet_min_train_loss_model_classification3d_array-{}.pth".format(
            save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_train_loss,  # self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_train_loss,
        )

    def save_model_valid_loss(self, n_epoch, save_path):
        self.lastmodel_valid_loss = "/data/data_Flair/{}-resnet_min_valid_loss_model_classification3d_array-{}.pth".format(
            save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_loss,  # self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_valid_loss,
        )

    def save_model_valid_ROC(self, n_epoch, save_path):
        self.lastmodel_valid_ROC = "/data/data_Flair/{}-resnet_max_valid_ROC_model_classification3d_array-{}.pth".format(
            save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_ROC,  # self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_valid_ROC,
        )

    def save_model_valid_fscore(self, n_epoch, save_path):
        self.lastmodel_valid_fscore = "/data/data_Flair/{}-resnet_max_valid_fscore_model_classification3d_array-{}.pth".format(
            save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_fscore,  # self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_valid_fscore,
        )


    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)
 

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader(random_state):
    # df_train, df_valid, dummies_train, products_train, y_train, dummies_valid, products_valid, y_valid = ran_state(
    #     random_state)
    train_transforms = Compose(
        [
            # AsChannelFirst(),

            Rand3DElastic(  # mode=("bilinear", "nearest"),
                prob=0.25,
                sigma_range=(5, 7),
                magnitude_range=(50, 150),
                spatial_size=(128, 128, 128),
                # translate_range=(2, 2, 2),
                # rotate_range=(np.pi/36, np.pi/36, np.pi),
                # scale_range=(0.15, 0.15, 0.15),
                padding_mode="zeros"),

            RandAffine(  # mode=("bilinear", "nearest"),
                prob=0.25,
                spatial_size=(128, 128, 128),
                translate_range=(0.5, 0.5, 0.5),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="zeros"),
            # Orientation(axcodes="PLI"),
            # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            RandSpatialCrop(roi_size=(96, 96, 96)),
            Resize(spatial_size=(128, 128, 128)),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            # RandFlip(spatial_axis=0, prob=0.5),
            ScaleIntensity(),
            EnsureType(),
        ]
    )
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
    train_df = pd.read_csv("{}/Overall_Survival_1year.csv".format(data_directory))
    # display(train_df)
    X = train_df["Patient"].values
    y = train_df["OS 1 year"]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    train_loader_list = []
    valid_loader_list = []
    for train_index, valid_index in rskf.split(X, y):
        print("Train: {}".format(train_index), "Valid: {}".format(valid_index))
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        dummies_train = pd.get_dummies(y_train)  # Classification
        products_train = dummies_train.columns
        y_train_dummies = dummies_train.values
        dummies_valid = pd.get_dummies(y_valid)  # Classification
        products_valid = dummies_valid.columns
        y_valid_dummies = dummies_valid.values

        train_data_retriever = Dataset(
            paths=X_train,
            targets=y_train_dummies,
            norm_set_of_files=norm_set_of_files,
            split="Pcnls_baseline",
            transforms=train_transforms
        )
        valid_data_retriever = Dataset(
            paths=X_valid,
            targets=y_valid_dummies,
            norm_set_of_files=norm_set_of_files,
            split="Pcnls_baseline",
            transforms=valid_transforms
        )

        train_loader = DataLoader(  # torch_data.DataLoader(
            train_data_retriever,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate
        )
        valid_loader = DataLoader(  # torch_data.DataLoader(
            valid_data_retriever,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate
        )

        train_loader_list.append(train_loader)
        valid_loader_list.append(valid_loader)

    return train_loader_list, valid_loader_list


def train_all_type(norm_set_of_files=norm_set_of_files, random_state=random_state, mri_type="all"):
    train_loader_list, valid_loader_list = loader(random_state)
    # print(relationship.shape)
    # model = Model()
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=3)#.to(device)
    # model = monai.networks.nets.resnet.resnet50(n_input_channels=2, n_classes=2)#.to(device)
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # criterion = torch.nn.CrossEntropyLoss()#torch_functional.binary_cross_entropy_with_logits
    criterion = torch.nn.LogSoftmax(dim=-1)
    # trade_off = 1.3

    trainer = Trainer(
        model,
        device,
        criterion,
        optimizer,
        scheduler,
        random_state,
    )

    history_train_loss = trainer.fit_train_loss(
        250,
        train_loader_list,
        valid_loader_list,
        f"{mri_type}",
        50,
    )


    history_valid_ROC = trainer.fit_valid_ROC(
        250,
        train_loader_list,
        valid_loader_list,
        f"{mri_type}",
        50,
    )

    return history_train_loss,  history_valid_ROC,  trainer.lastmodel_train_loss,  trainer.lastmodel_valid_ROC








