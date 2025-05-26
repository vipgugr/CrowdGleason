import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import data
import train
import tensorflow as tf
import gpflow


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import pandas as pd

# Hyperparameters
n_epochs = 50
n_inducings = [512] # [32, 64, 128, 256, 512, 1024]
batch_size = 128
runs = 5
K = 4

# Experiment
train_names = ['grxcr'] 
val_names = ['vlc', 'grxmv']

for run_ann in range(10):
    for n_annotators in range(1, 7):
        for n_inducing in n_inducings:
            for ix_run in range(runs):
                for train_name in train_names:
                    for val_name in val_names:
                        # Save_path
                        save_path = 'experiments/{}/{}_{}/{}/{}_annotators/run_{}'.format(ix_run,train_name,val_name,n_inducing,n_annotators,run_ann)
                        
                        
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        # Crowdsourcing
                        crowd = False
                        if train_name == 'grxcr':
                            crowd = True

                        # Load and select data
                        data_loaded = data.load_data(ix_run, n_annotators)
                        data_loaded.select(train_name, val_name)

                        # Train dataset
                        X_tr = data_loaded.X_train

                        if crowd:
                            y_tr_ev = data_loaded.grx.train_MV
                            y_tr = data_loaded.y_train
                        else:
                            y_tr = data_loaded.y_train
                            y_tr_ev = y_tr

                        # Validation dataset
                        X_vl = data_loaded.X_val
                        y_vl= data_loaded.y_val

                        # Model definition and train
                        iters_per_epoch = len(X_tr)//batch_size
                        model = train.create_setup(X_tr, y_tr, 2.0, 2.0, batch_size, n_inducing, K, crowd, False)
                        
                        best_model, best_val \
                            = train.run_adam(model, n_epochs, iters_per_epoch, X_tr,
                                                y_tr_ev, X_vl, y_vl, save_path)


                        print('The best model in val obtained\n ' + best_val[0] + ': ' + str(best_val[1]))

                        #############################################
                        ##############       TEST       #############
                        #############################################
                        results_dict = {"data": [], "f1": [], "acc": [], "kap": [], "kap2": []}

                        # SICAP test
                        X_ts = data_loaded.sicap.X_test
                        y_ts = data_loaded.sicap.y_test
                        results = train.evaluate(best_model,X_ts,y_ts)
                        print("SICAP:\n", results)
                        for key in results.keys():
                            results_dict[key].append(results[key])
                        results_dict['data'].append('vlc')

                        X_ts = data_loaded.grx.X_test
                        y_ts = data_loaded.grx.test_EM
                        results = train.evaluate(best_model,X_ts,y_ts)
                        print("GRX-EM:\n",results)
                        for key in results.keys():
                            results_dict[key].append(results[key])
                        results_dict['data'].append('GRX-EM')

                        y_ts = data_loaded.grx.test_MV
                        results = train.evaluate(best_model,X_ts,y_ts)
                        print("GRX-MV:\n",results)
                        for key in results.keys():
                            results_dict[key].append(results[key])
                        results_dict['data'].append('GRX-MV')

                        # The official CrowdGleason (at zenodo) already has consensus labels so this further filter out is not necessary
                        indexes = data_loaded.grx.index_consesus
                        X_ts_consensus = X_ts[indexes]
                        y_ts_consensus = np.array(y_ts)[indexes]
                        results = train.evaluate(best_model,X_ts_consensus,y_ts_consensus)
                        print("Consensus:\n",results)
                        for key in results.keys():
                            results_dict[key].append(results[key])
                        results_dict['data'].append('consensus')

                        for i in range(7):
                            y_ts = data_loaded.grx.y_test[:,i]
                            results = train.evaluate(best_model,X_ts,y_ts)
                            print("Marker" + str(i+1) + ":\n",results)
                            for key in results.keys():
                                results_dict[key].append(results[key])
                            results_dict['data'].append('Marker '+ str(i+1))

                        results_df = pd.DataFrame(results_dict)
                        results_df.to_csv(save_path + "results.csv", index=False)
