import numpy as np
import tensorflow as tf

from gpflow import kullback_leiblers, features
from gpflow import settings
from gpflow import transforms
from gpflow.conditionals import conditional, Kuu
from gpflow.decors import params_as_tensors
from gpflow.models.model import GPModel
from gpflow.params import DataHolder, Minibatch, Parameter

import time

class SVGPMix(GPModel):

    def __init__(self, X, ytrue, Y,
                 kern, likelihood, mean_function=None,
                 feat=None, Z=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 num_data=None,
                 num_latent = None,
                 q_unn = None,
                 q_mu=None, q_sqrt=None,
                 alpha = None, alpha_tilde = None, **kwargs):
        """
        - X is a data matrix, size N x D.
        - ytrue contains the index and the true labels of a small subset of the training set.
        - Y contains the annotations. List of matrices with 2 columns, gathering pairs (annotator, annotation). It is not a NxA matrix (with A the number of annotators) because an annotator can an label an instance more than once.
        - kern, likelihood, mean_function are appropriate GPflow objects.
        - feat and Z define the inducing point locations, usually feat=None and Z is size M x D
        - q_diag, boolean indicating whether posterior covariance must be diagonal
        - withen, boolean indicating whether a whitened representation of the inducing points is used (this only affects the internal representations and computations involving inducing points, but not the final result)
        - minibatch_size, if not None, turns on mini-batching with that size
        - num_data is the total number of observations (only relevant when feeding in external minibatches, if None then it defaults to X.shape[0])
        - num_latent is the number of latent GPs to be used
        - q_mu: (initialization for the) mean of the normal distribution used as approximate posterior q(U). Size M x K.
        - q_sqrt: (initialization for the) cholesky factor of the variance of the normal distribution q(U). Size K x M x M (if q_diag=False) and M x K (if q_diag=True).
        - alpha/alpha_tilde: (initializations for the) parameters that define the prior p(R) and the posterior q(R). Both have size A x K x K. Recall that alpha_tilde is estimated and alpha is fixed.
        """
        if minibatch_size is None:
            X = DataHolder(X)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
        class_keys = np.unique(np.concatenate((ytrue[:,1],np.concatenate([y[:,1] for y in Y[1]])))) # unique identifiers for classes
        num_classes = len(class_keys)
        num_latent = num_latent or num_classes
        GPModel.__init__(self, X, None, kern, likelihood, mean_function, num_latent, **kwargs)
        self.class_keys = class_keys
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.annot_keys = np.unique(np.concatenate([y[:,0] for y in Y[1]])) # unique identifiers for annotators
        self.num_annotators = len(self.annot_keys)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_inducing = len(self.feature)
        # MIXTO ###############################################################        
        self.ytrue = ytrue
        self.ind_O = ytrue[:,0]
        self.ind_U = np.delete(np.arange(self.num_data),self.ind_O)
        self.ind_OU = np.concatenate((self.ind_O,self.ind_U))
        self.perm_OU = np.argsort(self.ind_OU)
        #######################################################################
        
        #### Initializing minibatches or placeholders (and the associated idxs_mb to slice q_unn, which seemingly cannot be wrapped in a minibatch or placeholder, maybe because it is already a gpflow Parameter).
        startTime = time.time()
        Y_idxs = np.array([np.stack((np.array([np.flatnonzero(v==self.annot_keys)[0] for v in y[:,0]]),
                                     np.array([np.flatnonzero(v==self.class_keys)[0] for v in y[:,1]])), axis=1) for y in Y[1]]) # same as Y but contains indexes over annot_keys and class_keys
        
        # MIXTO ############################################################### 
        self.idx_cr = np.array(Y[0].copy(),dtype= int)
        Y_idxs_U = np.array([Y_idxs[self.idx_cr==y] for y in self.ind_U])
        if (Y_idxs_U.shape[0]>0):
            Y_idxs_U = Y_idxs_U[:,0]
        #######################################################################
        
        S = np.max([v.shape[0] for v in Y_idxs]) # Maximum number of annotations for a single instance
        
        # MIXTO ###############################################################
        #Y_idxs_cr = np.array([np.concatenate((y,-1*np.ones((S-y.shape[0],2))),axis=0) for y in Y_idxs]).astype(np.int16) # Padding Y_idxs with -1 to create a NxSx2 array.
        aux = np.array([self.num_annotators,0])
        Y_idxs_cr_aux = np.array([np.concatenate((y,np.tile(aux,(S-y.shape[0],1))),axis=0) for y in Y_idxs]).astype(np.int16) # Padding Y_idxs with -1 to create a NxSx2 array.
        #Y_idxs_cr_aux = np.array([np.concatenate((y,-1*np.ones((S-y.shape[0],2))),axis=0) for y in Y_idxs]).astype(np.int16) # Padding Y_idxs with -1 to create a NxSx2 array.
        #aux = np.tile(aux,(self.num_data,S,1))
        Y_idxs_cr = np.tile(aux,(self.num_data,S,1))
        #Y_idxs_cr_old = -1*np.ones((self.num_data,S,2),dtype = int)
        #print(aux.shape)
        #print(Y_idxs_cr.shape)        
        Y_idxs_cr[self.idx_cr,:,:] = Y_idxs_cr_aux
        #Y_idxs_cr_old[self.idx_cr,:,:] = Y_idxs_cr_aux
        #######################################################################
        if minibatch_size is None:
            self.Y_idxs_cr = DataHolder(Y_idxs_cr)
            self.idxs_mb = DataHolder(np.arange(self.num_data))
        else:
            self.Y_idxs_cr = Minibatch(Y_idxs_cr, batch_size=minibatch_size, seed=0)
            self.idxs_mb = Minibatch(np.arange(self.num_data), batch_size=minibatch_size, seed=0)
        print("Time taken in Y_idxs creation:", time.time()-startTime)
        
        #### Initializing the approximate posterior q(z). q_unn is NxK, and constrained to be positive. "_unn" denotes unnormalized: rows must be normalized to obtain the posterior q(z). We could have also implemented a gpflow transform that constrains sum to 1 by rows. Notice that q_unn initialization is based on the annotations.
        startTime = time.time()
        # MIXTO ###############################################################  
        #q_unn = np.array([np.bincount(y[:,1], minlength=self.num_classes) for y in Y_idxs]) 
        
        if q_unn is None:
            q_unn = np.array([np.bincount(y[:,1], minlength=self.num_classes) for y in Y_idxs_U])
        #######################################################################
            if q_unn.shape[0]!= 0:
                q_unn = q_unn + np.ones(q_unn.shape)
                q_unn = q_unn/np.sum(q_unn,axis=1,keepdims=True)
        # MIXTO ###############################################################  
        else:
            q_unn = q_unn[self.ind_U,:]
        self.q_unn = Parameter(q_unn,transform=transforms.positive) # N x K
        # MIXTO ############################################################### 
        if (self.num_classes==1):
            Q_unn = np.zeros([self.num_data,2])
        else:
            Q_unn = np.zeros([self.num_data,self.num_classes])
        if q_unn.shape[0]!= 0:
            Q_unn[self.ind_U] = np.copy(q_unn)
        Q_unn[self.ind_O,ytrue[:,1].astype(np.int16)] = 1
        #######################################################################
        print("Time taken in q_unn initialization:", time.time()-startTime)
        
        #### Initializing alpha (fixed) and alpha_tilde (trainable). Both have size AxKxK
        if alpha is None:
            alpha = np.ones((self.num_annotators,self.num_classes,self.num_classes), dtype=settings.float_type)
        self.alpha = Parameter(alpha, transform=transforms.positive, trainable=False)
        startTime = time.time()
        #######################################################################
        #alpha_tilde = self._init_behaviors(q_unn, Y_idxs)
        alpha_tilde = self._init_behaviors(Q_unn[self.idx_cr], Y_idxs) #MIXTO
        #######################################################################
        print("Time taken in alpha_tilde initialization:", time.time()-startTime)
        self.alpha_tilde = Parameter(alpha_tilde,transform=transforms.positive)
        #### Initializing the variational parameters
        self._init_variational_parameters(q_mu, q_sqrt)
        
    def _init_variational_parameters(self, q_mu, q_sqrt):
        q_mu = np.zeros((self.num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x K

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((self.num_inducing, self.num_latent), dtype=settings.float_type), transform=transforms.positive)  # M x K
            else:
                q_sqrt = np.array([np.eye(self.num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(self.num_inducing, self.num_latent))  # K x M x M
        else:
            if self.q_diag:
                assert q_sqrt.ndim == 2
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x K
            else:
                assert q_sqrt.ndim == 3
                self.q_sqrt = Parameter(q_sqrt,transform=transforms.LowerTriangular(self.num_inducing, self.num_classes))  # K x M x M

    def _init_behaviors(self, probs, Y_idxs):
        alpha_tilde = np.ones((self.num_annotators,self.num_classes,self.num_classes))/self.num_classes
        counts = np.ones((self.num_annotators,self.num_classes))
        for n in range(len(Y_idxs)):
            for a,c in zip(Y_idxs[n][:,0], Y_idxs[n][:,1]):
                alpha_tilde[a,c,:] += probs[n,:]
                counts[a,c] += 1
        alpha_tilde=alpha_tilde/counts[:,:,None]
        alpha_tilde = (counts/np.sum(counts,axis=1,keepdims=True))[:,:,None]*alpha_tilde
        return alpha_tilde/np.sum(alpha_tilde,axis=1,keepdims=True)

    @params_as_tensors
    def build_prior_KL(self): # Computes KL div between q(U) and p(U) [4th term in eq.(15) in the paper]
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)
        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def build_annot_KL(self): # Computes KL div between q(R) and p(R) [5th term in eq.(15) in the paper]
        alpha_diff = self.alpha_tilde-self.alpha
        KL_annot=(tf.reduce_sum(tf.multiply(alpha_diff,tf.digamma(self.alpha_tilde)))-
        tf.reduce_sum(tf.digamma(tf.reduce_sum(self.alpha_tilde,1))*tf.reduce_sum(alpha_diff,1))+
        tf.reduce_sum(tf.lbeta(tf.matrix_transpose(self.alpha))
        -tf.lbeta(tf.matrix_transpose(self.alpha_tilde))))
        return KL_annot

    @params_as_tensors
    def _build_likelihood(self): # It returns the ELBO in eq.(15) in the paper

        KL = self.build_prior_KL() # [4th term in eq.(15) in the paper]
        KL_annot = self.build_annot_KL() # [5th term in eq.(15) in the paper]

        #### ENTROPY OF Q COMPONENT [3rd term in eq.(15) in the paper]
        # Mixto ###############################################################
        if self.q_unn.shape[0]!= 0:
            Q = tf.concat([tf.one_hot(self.ytrue[:,1].astype(np.int32),self.num_classes,dtype=tf.float64),self.q_unn],axis=0)
        else:
            Q = tf.one_hot(self.ytrue[:,1].astype(np.int32),self.num_classes,dtype=tf.float64)    
        Q = tf.gather(Q,self.perm_OU)
        #q_unn_mb = tf.gather(self.q_unn,self.idxs_mb)       # N x K 
        q_unn_mb = tf.gather(Q,self.idxs_mb)              # Mixto
        #######################################################################
        q_mb = tf.divide(q_unn_mb, tf.reduce_sum(q_unn_mb, axis=1, keepdims=True)) # N x K
        qentComp = tf.reduce_sum(tf.log(tf.pow(q_mb,q_mb)))
        #qentComp = tf.reduce_sum(tf.multiply(q_mb,tf.log(q_mb)))
                
        #### LIKELIHOOD COMPONENT [2nd term in eq.(15) in the paper]
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)
        tensors_list = [self.likelihood.variational_expectations(fmean, fvar, c*tf.ones((tf.shape(self.X)[0],1),dtype=tf.int32)) for c in np.arange(self.num_classes)]
        tnsr_lik = tf.concat(tensors_list,-1)  # NxK
        lhoodComp = tf.reduce_sum(tf.multiply(q_mb,tnsr_lik))

        #### CROWDSOURCING COMPONENT [1st term in eq.(15) in the paper]
        expect_log = tf.digamma(self.alpha_tilde)-tf.digamma(tf.reduce_sum(self.alpha_tilde,1,keepdims=True)) # A x K x K
        expect_log = tf.concat([expect_log,tf.zeros([1,self.num_classes,self.num_classes],tf.float64)],0)        
        tnsr_expCrow = tf.gather_nd(expect_log, tf.cast(self.Y_idxs_cr, tf.int32))
        crComp = tf.reduce_sum(tf.multiply(tnsr_expCrow, tf.expand_dims(q_mb,1)))

        scale = tf.cast(self.num_data, settings.float_type)/tf.cast(tf.shape(self.X)[0], settings.float_type)
        self.decomp = [lhoodComp,crComp,qentComp,KL,KL_annot,scale]
        return ((lhoodComp+crComp-qentComp)*scale-KL-KL_annot)
                

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False): # Compute the approximate posterior in the entries Xnew 
        mu, var = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov, white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var
