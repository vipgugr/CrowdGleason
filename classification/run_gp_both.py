import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import data
import train
import tensorflow as tf
import gpflow


#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np
import pandas as pd

# Hyperparameters
n_epochs = 50
n_inducings = [512]
batch_size = 128
K = 4
runs = 5

# Experiment
for run in range(runs):
    for n_inducing in n_inducings:
        # Save_path
        save_path = 'experiments/{}/{}/{}/'.format(run,'both_vlc_grxmv',n_inducing)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Load and select data
        data_loaded = data.load_data(run)
        data_loaded.select('grxmv', 'vlc')

        # Train dataset
        X_tr = np.concatenate((data_loaded.X_train, data_loaded.sicap.X_train))
        y_tr = np.concatenate((data_loaded.y_train, data_loaded.sicap.y_train))
        y_tr_ev = y_tr

        print("DATA: ", X_tr.shape, y_tr.shape)

        # Validation dataset
        X_vl = data_loaded.X_val
        y_vl= data_loaded.y_val

        # Model definition and train
        iters_per_epoch = len(X_tr)//batch_size
        model = train.create_setup(X_tr, y_tr, 2.0, 2.0, batch_size, n_inducing, K, False, False)

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
