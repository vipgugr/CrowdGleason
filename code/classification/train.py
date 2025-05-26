import gpflow
import tensorflow as tf
import numpy as np
import pickle
import os
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gpflow
from scipy.cluster.vq import kmeans2
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, accuracy_score, classification_report
from typing import Tuple

from crowd_models.svgpcr import SVGPCR
from crowd_models.svgpmix import SVGPMix

figure_of_merit = 'kap2'


def batch_prediction_multi_classification(model, prediction_model, X, S):
    n_batches = max(int(X.shape[0]/50.), 1)
    prob_list = []
    for X_batch in zip(np.array_split(X, n_batches)):
        X_batch = np.array(X_batch)
        X_batch =  np.squeeze(X_batch, axis=0)
        prob = prediction_model(model, X_batch, S)
        prob_list.append(prob)
    prob = np.concatenate(prob_list, 0)
    return prob

def prediction_multi_dgp(model, X_batch, S):
    m, v = model.predict_y(X_batch, S)
    prob = np.average(m,0)
    return prob

def create_setup(X, y, lengthscale, variance, minibatch_size, num_inducing, K, crowd, deep=False):
    gpflow.reset_default_graph_and_session()
    X = X.astype('float64')
    Z = kmeans2(X,num_inducing,minit='points')[0]     
    if crowd: 

        model = SVGPCR(X=X, Y=y, kern=gpflow.kernels.RBF(X.shape[1], lengthscales=lengthscale, variance=variance),
                        likelihood=gpflow.likelihoods.SoftMax(K),
                        Z=Z,
                        num_latent=K,
                        num_data=len(X),
                        minibatch_size=minibatch_size
                        )
    else:
        Y = np.array(y).astype('int64')
        model = gpflow.models.SVGP(X=X, Y=Y, kern=gpflow.kernels.RBF(X.shape[1], lengthscales=lengthscale, variance=variance),
                            likelihood=gpflow.likelihoods.SoftMax(K),
                            Z=Z,
                            num_latent=K,
                            num_data=len(X),
                            minibatch_size=minibatch_size
                            )
    return model

def create_setup_svgpmix(X, y_cr, z, lengthscale, variance, minibatch_size, num_inducing, K):
    gpflow.reset_default_graph_and_session()
    X = X.astype('float64')
    Z = kmeans2(X,num_inducing,minit='points')[0]     
    model = SVGPMix(X=X, Y=y_cr, ytrue=np.array(z), kern=gpflow.kernels.RBF(X.shape[1], lengthscales=lengthscale, variance=variance),
                            likelihood=gpflow.likelihoods.SoftMax(K),
                            Z=Z,
                            num_latent=K,
                            num_data=len(X),
                            minibatch_size=minibatch_size
                            )

    return model

def run_adam(model, n_epochs, iter_per_epochs, X_tr=None, y_tr=None, X_vl=None, y_vl=None, save_path=None):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    optimizer = gpflow.training.AdamOptimizer(0.01)
    
    optimizer_tensor = optimizer.make_optimize_tensor(model)
    session = gpflow.get_default_session()

    train_metrics = {"f1": [], "acc": [], "kap": [], "kap2": []}
    val_metrics = {"f1": [], "acc": [], "kap": [], "kap2": []}

    best_val_metric = 0

    for epoch in range(n_epochs):
        print('    Epoch: {:2d} / {}'.format(epoch+1, n_epochs))
        
        for _  in range(iter_per_epochs):
             session.run(optimizer_tensor)
        model.anchor(session)
        train_step_metrics = evaluate(model, X_tr, y_tr)
        val_step_metrics = evaluate(model, X_vl, y_vl)
        for key in train_metrics.keys():
            train_metrics[key].append(train_step_metrics[key])
            val_metrics[key].append(val_step_metrics[key])
        print('ELBO: ', np.mean(np.array([model.compute_log_likelihood() for a in range(10)])))

        # Save metric
        if val_metrics[figure_of_merit][-1] > best_val_metric:
            print('Best model saved!')
            best_val_metric = val_metrics[figure_of_merit][-1]
            best_params = model.read_trainables()
            with open(save_path + 'best_svgp.pickle', 'wb') as handle:
                pickle.dump(best_params, handle)

    # Load best model
    model.assign(best_params)

    # Save logs
    logs = {'train': train_metrics, 'val': val_metrics}
    with open(save_path + 'logs.pickle', 'wb') as handle:
                pickle.dump(logs, handle)

    return model, (figure_of_merit, best_val_metric)

####### Functions to predict and evaluate

def predict(model,X_ts_nor):
    y_pred,_ = model.predict_y(X_ts_nor)  # y_pred is N_ts,K (test probabilities, adds 1 by rows)
    return y_pred

def evaluate(model,X_ts_nor,y_test):
    y_test = np.array(y_test)
    indexes = (y_test!=-1)
    y_test = y_test[indexes]

    y_pred_ = predict(model,X_ts_nor)
    y_pred = np.argmax(y_pred_,axis=1)
    y_pred = y_pred[indexes]
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_test,y_pred, average='macro')
    kap = cohen_kappa_score(y_test, y_pred)
    kap2 = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    row_sums = y_pred_.sum(axis=1)
    y_pred_ = y_pred_ / row_sums[:, np.newaxis]

    print(classification_report(y_test, y_pred))
    return {'f1':f1,'acc':acc,'kap':kap,'kap2':kap2}

def evaluate_crowd(model,X_ts_nor,y_test,cm):
    y_test = np.array(y_test)
    indexes = (y_test!=-1)
    y_test = y_test[indexes]

    y_pred_ = predict(model,X_ts_nor)
    y_pred_ann = cm @ y_pred_.T 
    y_pred_ann = y_pred_ann.T
    y_pred = np.argmax(y_pred_ann,axis=1)
    y_pred = y_pred[indexes]
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_test,y_pred, average='macro')
    kap = cohen_kappa_score(y_test, y_pred)
    kap2 = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    print(classification_report(y_test, y_pred))
    return {'f1':f1,'acc':acc,'kap':kap,'kap2':kap2} 