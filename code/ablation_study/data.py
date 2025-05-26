import pickle
import numpy as np
import pandas as pd
import random


expertise_annotators = {'junior': np.array([0,1,2]), 'senior': np.array([3,4,5,6])}

def unpack_data(data):
    return data['features'], data['label_list'], data['names']

def unpack_data_grx_train(data):
    return data['features'], np.array(data['label_list']), data['names'], data['MV'], data['EM']

def unpack_data_grx_test(data):
    return data['features'], np.array(data['label_list']), data['names'], data['MV'], data['EM'], data['GT']

def process_labels_cr(labels):
    labels_cr = np.zeros((labels.shape[0], labels.shape[1], 2))
    labels_mask = np.zeros((labels.shape[0], labels.shape[1]))
    for ix, x in enumerate(labels):
        for iy, y in enumerate(x):
            if y != -1:
                labels_cr[ix, iy, 0] = iy   
                labels_cr[ix, iy, 1] = y
                labels_mask[ix, iy] = 1
            else:
                labels_cr[ix, iy, 0] = -1   
                labels_cr[ix, iy, 1] = -1
                labels_mask[ix, iy] = 0
    return labels_cr, labels_mask

def process_labels_cr_v1(labels, n=None, expertise=None):
    if n is not None:
        sampled_annotators = random.sample(range(labels.shape[1]), k=n)
        labels = labels[:,sampled_annotators]

        print(sampled_annotators, labels.shape)
    elif expertise is not None:
        print(expertise, expertise_annotators)
        sampled_annotators = expertise_annotators[expertise]
        labels = labels[:,sampled_annotators]
        print(labels.shape)
    labels_cr = []
    labels_mask = np.ones((labels.shape[0], labels.shape[1]))
    instance_mask = np.zeros((labels.shape[0],1))

    for ix, x in enumerate(labels):
        labels_im = []
        for iy, y in enumerate(x):
            if y != -1:
                labels_im.append([iy, y]) 
            else:
                labels_mask[ix, iy] = 0

        if labels_mask[ix,:].sum() == 0:
            instance_mask[ix] = 0
        else:
            instance_mask[ix] = 1
        #print(labels_mask[ix], instance_mask[ix], labels_im)
        labels_cr.append(np.array(labels_im))
    instance_mask = np.array(instance_mask, dtype='bool') 
    #print(instance_mask)

    #print(labels_cr)
    return labels_cr, labels_mask, instance_mask

class load_data:
    def __init__(self, run, n_annotators=None, expertise=None):
        print("Load data")
        self.run = run
        self.n_annotators = n_annotators
        self.expertise = expertise

    def select(self, train, val):
        self.sicap = SICAP_data(self.run)
        self.grx = grx_data(self.run, self.n_annotators, self.expertise)

        if train == 'grxem':
            self.X_train = self.grx.X_train
            self.y_train = self.grx.train_EM
        elif train == 'grxmv':
            self.X_train = self.grx.X_train
            self.y_train = self.grx.train_MV
        elif train == 'grxcr':
            self.X_train = self.grx.X_train
            self.y_train = self.grx.y_train
        elif train == 'vlc':
            self.X_train = self.sicap.X_train
            self.y_train = self.sicap.y_train
        else:
            raise ("Set not found")

        # Validation
        if val == 'grxem':
            self.X_val = self.grx.X_val
            self.y_val = self.grx.val_EM
        elif val == 'grxmv':
            self.X_val = self.grx.X_val
            self.y_val = self.grx.val_MV
        elif val == 'grxcr':
            raise NotImplementedError
        elif val == 'vlc':
            self.X_val = self.sicap.X_val
            self.y_val = self.sicap.y_val
        
        self.m, self.std = self.X_train.mean(0), self.X_train.std(0)

        print("Train: ", train, self.X_train.shape, "\n Val: ", val, self.X_val.shape)

        self.X_train = self._norm(self.X_train)
        self.X_val = self._norm(self.X_val)

        self.sicap.X_test = self._norm(self.sicap.X_test)
        self.grx.X_test = self._norm(self.grx.X_test)

    def _norm(self, data):
        return (data-self.m)/self.std
        
class SICAP_data:
    def __init__(self, run=0):

        # Normalize

        # load data
        path_features = '../train_feat_ex/save/{}/sicap_norm_compatible.pickle'.format(run)
        with open(path_features, 'rb') as fp:
            sicap_data = pickle.load(fp)

        # training
        print(sicap_data.keys())
        train = sicap_data['train']
        self.X_train, self.y_train, self.train_names = unpack_data(train)

        # validation
        val = sicap_data['val']
        self.X_val, self.y_val, self.val_names = unpack_data(val)

        # test
        test = sicap_data['test']
        self.X_test, self.y_test, self.test_names = unpack_data(test)

        print('Train: ', self.X_train.shape)
        print('Val: ', self.X_val.shape)
        print('Test: ', self.X_test.shape)


class grx_data:
    def __init__(self, run=0, n_annotators=None, expertise=None):

        # load data

        path_features = '../train_feat_ex/save/{}/features_grx_norm_compatible.pickle'.format(run)
        with open(path_features, 'rb') as fp:
            grx_data = pickle.load(fp)

        # training
        print(grx_data.keys())
        train = grx_data['train']
        self.X_train, self.y_train, self.train_names, \
            self.train_MV, self.train_EM  = unpack_data_grx_train(train)
        
        self.y_train, self.train_mask, self.instance_mask = process_labels_cr_v1(self.y_train, n_annotators, expertise)
        mask = self.instance_mask.squeeze(-1)
        indices = np.where(mask)[0]
        # Asegurarse de que la máscara sea booleana
        self.X_train = self.X_train[indices]
        self.y_train = [self.y_train[i] for i in indices]
        # validation
        val = grx_data['val']
        self.X_val, self.y_val, self.val_names, \
            self.val_MV, self.val_EM = unpack_data_grx_train(val)

        # test
        test = grx_data['test']
        self.X_test, self.y_test, self.test_names, \
        self.test_MV, self.test_EM, self.test_GT = unpack_data_grx_test(test)

        self.index_consesus = (np.array(self.test_MV)==np.array(self.test_GT))

        # self.y_test_crowd = [x[:,i] for i in range(7) ]

        print('Train: ', self.X_train.shape)
        print('Val: ', self.X_val.shape)
        print('Test: ', self.X_test.shape)


def processing_agg(name, label_df):
    name = [x.split('.')[0] for x in name]
    name_df = pd.DataFrame({'Patch filename': name})
    return pd.merge(name_df, label_df, on='Patch filename', how='left')
                 
class grx_data_agg:
    def __init__(self, agg_method):

        # load data
        with open('../train_feat_ext/features/grx_norm_old.pickle', 'rb') as fp:
            grx_data = pickle.load(fp)

        # training
        print(grx_data.keys())
        train = grx_data['train']
        self.X_train, _, train_names, \
            _,_  = unpack_data_grx_train(train)
        
        label_df = pd.read_csv("/work/work_mik/crowdsourcing_JA/feat_extraction/labels_grx/train_agg.csv")
        merged_train = processing_agg(train_names, label_df)
        self.y_train = list(merged_train[agg_method].values)
        
        # validation
        val = grx_data['val']
        self.X_val, _, val_names, \
            _, _ = unpack_data_grx_train(val)
        label_df_val = pd.read_csv("/work/work_mik/crowdsourcing_JA/feat_extraction/labels_grx/val_agg.csv")
        merged_val = processing_agg(val_names, label_df_val)
        self.y_val = list(merged_val[agg_method].values)



        

        print('Train: ', self.X_train.shape, len(self.y_train))
        print('Val: ', self.X_val.shape, len(self.y_val))