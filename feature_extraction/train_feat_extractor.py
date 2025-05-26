import os
import sys
import numpy as np
import pandas as pd
import config.config_model as config

import pickle
# Graphic cards GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

import data
import model_builder
from sklearn.metrics import classification_report, cohen_kappa_score
import config.config_data2 as config_data

import torch
import torch.optim as optim
import torch.nn as nn

# Load and preprocess data
preprocessing = data.get_preprocessing()
augmentation = data.get_training_augmentation()


for exp in ['both_mv', 'grx_mv', 'vlc']:
	for run in range(5):
		# Define the directory path
		directory = f'save/{run}/'

		# Create the directory if it doesn't exist
		os.makedirs(directory, exist_ok=True)
	
		print(f'Run: {run}; Experiment: {exp}')

		if exp == 'grx_mv':
			train_df = config_data.grx_data["train_df_mv"]
			data_dir_train = config_data.grx_data["data_dir_train"]
			imbalance = False
			epochs = 12


		elif exp == 'vlc':
			train_df = config_data.sicap_data["train_df"]
			data_dir_train = config_data.sicap_data["data_dir"]
			epochs = config.epochs


		elif exp == 'both_mv':
			imbalance = False
			crowd = False
			data_dir_train_grx = config_data.grx_data["data_dir_train"]
			train_df_grx = config_data.grx_data["train_df_mv"]

			train_df_vlc = config_data.sicap_data["train_df"]
			data_dir_train_vlc = config_data.sicap_data["data_dir"]
			epochs = 12


		class_weights = torch.tensor([1.,1.,1.,1.])

		# Dataset
		if exp == 'both_mv':
			train_dataset = data.ProstateDataset_vlc_grxmv(train_df_grx,
			train_df_vlc, data_dir_train_grx, data_dir_train_vlc, augmentation=augmentation, preprocessing=preprocessing)

		else:
			train_dataset = data.ProstateDataset(train_df, data_dir_train, augmentation=augmentation, preprocessing=preprocessing)

		# Validation set
		if exp == 'grx_mv':
			val_df = config_data.grx_data["val_df_mv"]
			data_dir_val = config_data.grx_data["data_dir_train"]
		else:
			val_df = config_data.sicap_data["val_df"]
			data_dir_val = config_data.sicap_data["data_dir"]
		val_dataset = data.ProstateDataset(val_df, data_dir_val, preprocessing=preprocessing)

		# Test set
		test_df_grx = config_data.grx_data["test_df"]
		print(len(test_df_grx))
		data_dir_test_grx = config_data.grx_data["data_dir_test"]
		test_df_vlc = config_data.sicap_data["test_df"]
		data_dir_test_vlc = config_data.sicap_data["data_dir"]

		test_dataset_grx = data.ProstateDataset(test_df_grx, data_dir_test_grx, preprocessing=preprocessing)
		test_dataset_vlc= data.ProstateDataset(test_df_vlc, data_dir_test_vlc, preprocessing=preprocessing)

		dataloaders_dict = {}
		dataloaders_dict['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
		dataloaders_dict['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
		dataloaders_dict['test_grx'] = torch.utils.data.DataLoader(test_dataset_grx, batch_size=config.batch_size, shuffle=False, num_workers=0)
		dataloaders_dict['test_vlc'] = torch.utils.data.DataLoader(test_dataset_vlc, batch_size=config.batch_size, shuffle=False, num_workers=0)

		# Detect if we have a GPU available
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print('device! ', device)

		model_name = config.model

		
		model_ft = model_builder.initialize_model("resnet", num_classes=4)

		# Send the model to GPU
		model_ft = model_ft.to(device)

		# Observe that all parameters are being optimized

		params_to_update = model_ft.parameters()
		optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

		# Setup the loss fxn
		class_weights = class_weights.to(device)


		# Train and evaluate
		if config.mode == 'train':
			model_ft, hist = model_builder.train_model(model_ft, dataloaders_dict, class_weights, optimizer_ft, num_epochs=epochs, exp=exp, run=run, is_inception=(model_name=="inception"))
		if config.mode == 'evaluate':
			model_ft = torch.load(f'save/{run}/vlc_best_.pth', device)

		model_ft.eval()


		### Testing

		# GRX

		print("Consensus GRX: ")

		preds_test = []
		labels_test = []

		for inputs, labels, _, _ in dataloaders_dict['test_grx']:
			inputs = inputs.to(device)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer_ft.zero_grad()

			outputs = model_ft(inputs)

			_, preds = torch.max(outputs, 1)



			preds_test.extend(preds.cpu().detach().numpy())
			labels_test.extend(labels.cpu().detach().numpy())

		preds_test = np.array(preds_test)
		labels_test = np.array(labels_test)

		np.save(f'save/{run}/{exp}_pred_vlc.npy', preds_test)
		results = classification_report(labels_test, preds_test, digits=4, output_dict=True)
		results['kappa'] = cohen_kappa_score(labels_test, preds_test)
		# Specify the filename
		filename = f'save/{run}/{exp}_grx.pkl'

		# Open the file in binary write mode and use pickle.dump() to save the dictionary
		with open(filename, 'wb') as file:
			pickle.dump(results, file)

		# VLC
		print("Experts VLC: ")
		preds_test = []
		labels_test = []

		for inputs, labels, _, _ in dataloaders_dict['test_vlc']:
			inputs = inputs.to(device)
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer_ft.zero_grad()

			outputs = model_ft(inputs)

			_, preds = torch.max(outputs, 1)

			preds_test.extend(preds.cpu().detach().numpy())
			labels_test.extend(labels.cpu().detach().numpy())

		preds_test = np.array(preds_test)
		labels_test = np.array(labels_test)

		np.save(f'save/{run}/{exp}_pred_vlc.npy', preds_test)
		results = classification_report(labels_test, preds_test, digits=4, output_dict=True)
		results['kappa'] = cohen_kappa_score(labels_test, preds_test)
		# Specify the filename
		filename = f'save/{run}/{exp}_vlc.pkl'

		# Open the file in binary write mode and use pickle.dump() to save the dictionary
		with open(filename, 'wb') as file:
			pickle.dump(results, file)
