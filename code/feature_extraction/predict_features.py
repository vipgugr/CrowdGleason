import os# Graphic cards GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import numpy as np
import pickle

import data
preprocessing = data.get_preprocessing()

import config.config_data as config_data



runs = 5
for i in range(runs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_pretrained = torch.load(f'save/{i}/vlc_best.pth', device)

    class new_model(torch.nn.Module):
        def __init__(self, output_layer):
            super().__init__()
            self.output_layer = output_layer
            self.pretrained = model_pretrained
            self.children_list = []
            for n,c in self.pretrained.named_children():
                self.children_list.append(c)
                if n == self.output_layer:
                    print("Feature layer!: ", n)
                    break

            self.net = torch.nn.Sequential(*self.children_list)
            self.pretrained = None
            
        def forward(self,x):
            x = self.net(x)
            return x

    model = new_model('avgpool')
    model.eval()

    
    print("SICAP - run: ", i) #### SICAP Dataset
    sicap = {}

    #Train
    train_df = config_data.sicap_data["train_df"]
    data_dir_train = config_data.sicap_data["data_dir"]

    # Validation set
    val_df = config_data.sicap_data["val_df"]
    data_dir_val = config_data.sicap_data["data_dir"]

    # Test set
    test_df_vlc = config_data.sicap_data["test_df"]
    data_dir_test_vlc = config_data.sicap_data["data_dir"]

    # Dataset
    train_dataset = data.ProstateDataset(train_df, data_dir_train, preprocessing=preprocessing)
    val_dataset = data.ProstateDataset(val_df, data_dir_val, preprocessing=preprocessing)
    test_dataset_vlc= data.ProstateDataset(test_df_vlc, data_dir_test_vlc, preprocessing=preprocessing)

    dataloaders_dict = {}
    dataloaders_dict['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    dataloaders_dict['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    dataloaders_dict['test'] = torch.utils.data.DataLoader(test_dataset_vlc, batch_size=256, shuffle=False, num_workers=0)


    for set_ in dataloaders_dict.keys():
        print(set_)
        features_list = []
        label_list = []
        name_list = []
        dataloader = dataloaders_dict[set_]
        with torch.no_grad():
            for inputs, labels, name, _ in dataloader:
                inputs = inputs.to(device)
                y = model(inputs)
                features_list.extend(y.cpu().detach().numpy())
                label_list.extend(labels.cpu().detach().numpy())
                name_list.extend(name)

        features_array = np.array(features_list)
        features_array = features_array.squeeze(-1).squeeze(-1)

        features = {'features': features_array, 'names': name_list, 'label_list': label_list}
        
        sicap[set_] = features


    with open(f'save_jpg/{i}/features_sicap_norm.pickle', 'wb') as handle:
        pickle.dump(sicap, handle, protocol=4)


    # ----------------------------

    print("GRX") 
    grx = {}

    #Train
    train_df = config_data.grx_data["train_df"]
    data_dir_train = config_data.grx_data["data_dir"] + "train/"

    # Validation set
    val_df = config_data.grx_data["val_df"]
    data_dir_val = config_data.grx_data["data_dir"] + "train/"

    # Test set
    test_df = config_data.grx_data["test_df"]
    data_dir_test = config_data.grx_data["data_dir"] + "test/"

    # Dataset
    train_dataset = data.CrowdProstateDataset(train_df, data_dir_train, preprocessing=preprocessing)
    val_dataset = data.CrowdProstateDataset(val_df, data_dir_val, preprocessing=preprocessing)
    test_dataset = data.CrowdProstateDataset(test_df, data_dir_test, preprocessing=preprocessing, experts=True)

    dataloaders_dict = {}
    dataloaders_dict['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    dataloaders_dict['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    dataloaders_dict['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)


    for set_ in  ['train', 'val']:
        print(set_)
        features_list = []
        label_list = []
        MV_list = []
        EM_list = []
        name_list = []
        dataloader = dataloaders_dict[set_]
        with torch.no_grad():
            for inputs, labels, name, MV_label, EM_label in dataloader:
                inputs = inputs.to(device)
                y = model(inputs)
                features_list.extend(y.cpu().detach().numpy())
                label_list.extend(labels.cpu().detach().numpy())
                name_list.extend(name)
                MV_list.extend(MV_label.cpu().detach().numpy())
                EM_list.extend(EM_label.cpu().detach().numpy())

        features_array = np.array(features_list)
        features_array = features_array.squeeze(-1).squeeze(-1)

        features = {'features': features_array, 'names': name_list, 'label_list': label_list, 'MV': MV_list, 'EM': EM_list}
        
        grx[set_] = features


    for set_ in  ['test']:
        print(set_)
        features_list = []
        label_list = []
        MV_list = []
        EM_list = []
        GT_list = []
        name_list = []
        dataloader = dataloaders_dict[set_]
        with torch.no_grad():
            for inputs, labels, name, GT_label in dataloader:
                inputs = inputs.to(device)
                y = model(inputs)
                features_list.extend(y.cpu().detach().numpy())
                label_list.extend(labels.cpu().detach().numpy())
                name_list.extend(name)
                GT_list.extend(GT_label.cpu().detach().numpy())

        features_array = np.array(features_list)
        features_array = features_array.squeeze(-1).squeeze(-1)

        features = {'features': features_array, 'names': name_list, 'label_list': label_list, 'GT': GT_list}
        
        grx[set_] = features

    with open(f'save_jpg2/{i}/features_grx_norm.pickle', 'wb') as handle:
        pickle.dump(grx, handle, protocol=4)

