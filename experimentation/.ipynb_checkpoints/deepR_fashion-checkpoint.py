import numpy as np
from keras.datasets import mnist, fashion_mnist
import pandas as pd
from keras.layers import Input, Dense, concatenate 
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm
from sklearn.decomposition import PCA
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras import objectives
from keras.backend.tensorflow_backend import set_session
from scipy.sparse import eye as speye
from scipy.sparse.csgraph import laplacian
from sklearn.manifold.locally_linear import (
    barycenter_kneighbors_graph, null_space, LocallyLinearEmbedding)
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.neighbors import NearestNeighbors

from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh



class LLEC(LocallyLinearEmbedding):
    def __init__(self, n_neighbors=5, n_components=2, reg=1E-3,
                 eigen_solver='auto', tol=1E-6, max_iter=100,
                 method='standard', hessian_tol=1E-4, modified_tol=1E-12,
                 neighbors_algorithm='auto', random_state=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.random_state = random_state
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs

    def fit_transform(self, X, y=None):
        self._fit_transform(X)
        return self.embedding_
    
    def fit_W(self, X):
        self._fit_transform(X)
        return self.W.toarray()
    
    def _fit_transform(self, X):
        self.nbrs_ = NearestNeighbors(self.n_neighbors,
                                      algorithm=self.neighbors_algorithm,
                                      n_jobs=self.n_jobs)

        random_state = self.random_state
        self.nbrs_.fit(X)
        self.W = \
            locally_linear_embedding(
                self.nbrs_, self.n_neighbors, self.n_components,
                eigen_solver=self.eigen_solver, tol=self.tol,
                max_iter=self.max_iter, method=self.method,
                hessian_tol=self.hessian_tol, modified_tol=self.modified_tol,
                random_state=random_state, reg=self.reg, n_jobs=self.n_jobs)

def locally_linear_embedding(
        X, n_neighbors, n_components, reg=1e-3, eigen_solver='auto', tol=1e-6,
        max_iter=100, method='standard', hessian_tol=1E-4, modified_tol=1E-12,
        random_state=None, n_jobs=None):
    
    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if method not in ('standard', 'hessian', 'modified', 'ltsa'):
        raise ValueError("unrecognized method '%s'" % method)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = (eigen_solver != 'dense')

    if method == 'standard':
        W = barycenter_kneighbors_graph(nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=1)

    
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

    elif method == 'hessian':
        dp = n_components * (n_components + 1) // 2

        if n_neighbors <= n_components + dp:
            raise ValueError("for method='hessian', n_neighbors must be "
                             "greater than "
                             "[n_components * (n_components + 3) / 2]")

        neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1,
                                    return_distance=False)
        neighbors = neighbors[:, 1:]

        Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1

        M = np.zeros((N, N), dtype=np.float64)

        use_svd = (n_neighbors > d_in)

        for i in range(N):
            Gi = X[neighbors[i]]
            Gi -= Gi.mean(0)

            # build Hessian estimator
            if use_svd:
                U = svd(Gi, full_matrices=0)[0]
            else:
                Ci = np.dot(Gi, Gi.T)
                U = eigh(Ci)[1][:, ::-1]

            Yi[:, 1:1 + n_components] = U[:, :n_components]

            j = 1 + n_components
            for k in range(n_components):
                Yi[:, j:j + n_components - k] = (U[:, k:k + 1] *
                                                 U[:, k:n_components])
                j += n_components - k

            Q, R = qr(Yi)

            w = Q[:, n_components + 1:]
            S = w.sum(0)

            S[np.where(abs(S) < hessian_tol)] = 1
            w /= S

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(w, w.T)

        if M_sparse:
            M = csr_matrix(M)

    elif method == 'modified':
        if n_neighbors < n_components:
            raise ValueError("modified LLE requires "
                             "n_neighbors >= n_components")

        neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1,
                                    return_distance=False)
        neighbors = neighbors[:, 1:]

        # find the eigenvectors and eigenvalues of each local covariance
        # matrix. We want V[i] to be a [n_neighbors x n_neighbors] matrix,
        # where the columns are eigenvectors
        V = np.zeros((N, n_neighbors, n_neighbors))
        nev = min(d_in, n_neighbors)
        evals = np.zeros([N, nev])

        # choose the most efficient way to find the eigenvectors
        use_svd = (n_neighbors > d_in)

        if use_svd:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                V[i], evals[i], _ = svd(X_nbrs,
                                        full_matrices=True)
            evals **= 2
        else:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                C_nbrs = np.dot(X_nbrs, X_nbrs.T)
                evi, vi = eigh(C_nbrs)
                evals[i] = evi[::-1]
                V[i] = vi[:, ::-1]

        # find regularized weights: this is like normal LLE.
        # because we've already computed the SVD of each covariance matrix,
        # it's faster to use this rather than np.linalg.solve
        reg = 1E-3 * evals.sum(1)

        tmp = np.dot(V.transpose(0, 2, 1), np.ones(n_neighbors))
        tmp[:, :nev] /= evals + reg[:, None]
        tmp[:, nev:] /= reg[:, None]

        w_reg = np.zeros((N, n_neighbors))
        for i in range(N):
            w_reg[i] = np.dot(V[i], tmp[i])
        w_reg /= w_reg.sum(1)[:, None]

        # calculate eta: the median of the ratio of small to large eigenvalues
        # across the points.  This is used to determine s_i, below
        rho = evals[:, n_components:].sum(1) / evals[:, :n_components].sum(1)
        eta = np.median(rho)

        # find s_i, the size of the "almost null space" for each point:
        # this is the size of the largest set of eigenvalues
        # such that Sum[v; v in set]/Sum[v; v not in set] < eta
        s_range = np.zeros(N, dtype=int)
        evals_cumsum = stable_cumsum(evals, 1)
        eta_range = evals_cumsum[:, -1:] / evals_cumsum[:, :-1] - 1
        for i in range(N):
            s_range[i] = np.searchsorted(eta_range[i, ::-1], eta)
        s_range += n_neighbors - nev  # number of zero eigenvalues

        # Now calculate M.
        # This is the [N x N] matrix whose null space is the desired embedding
        M = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            s_i = s_range[i]

            # select bottom s_i eigenvectors and calculate alpha
            Vi = V[i, :, n_neighbors - s_i:]
            alpha_i = np.linalg.norm(Vi.sum(0)) / np.sqrt(s_i)

            # compute Householder matrix which satisfies
            #  Hi*Vi.T*ones(n_neighbors) = alpha_i*ones(s)
            # using prescription from paper
            h = np.full(s_i, alpha_i) - np.dot(Vi.T, np.ones(n_neighbors))

            norm_h = np.linalg.norm(h)
            if norm_h < modified_tol:
                h *= 0
            else:
                h /= norm_h

            # Householder matrix is
            #  >> Hi = np.identity(s_i) - 2*np.outer(h,h)
            # Then the weight matrix is
            #  >> Wi = np.dot(Vi,Hi) + (1-alpha_i) * w_reg[i,:,None]
            # We do this much more efficiently:
            Wi = (Vi - 2 * np.outer(np.dot(Vi, h), h) +
                  (1 - alpha_i) * w_reg[i, :, None])

            # Update M as follows:
            # >> W_hat = np.zeros( (N,s_i) )
            # >> W_hat[neighbors[i],:] = Wi
            # >> W_hat[i] -= 1
            # >> M += np.dot(W_hat,W_hat.T)
            # We can do this much more efficiently:
            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(Wi, Wi.T)
            Wi_sum1 = Wi.sum(1)
            M[i, neighbors[i]] -= Wi_sum1
            M[neighbors[i], i] -= Wi_sum1
            M[i, i] += s_i

        if M_sparse:
            M = csr_matrix(M)

    elif method == 'ltsa':
        neighbors = nbrs.kneighbors(X, n_neighbors=n_neighbors + 1,
                                    return_distance=False)
        neighbors = neighbors[:, 1:]

        M = np.zeros((N, N))

        use_svd = (n_neighbors > d_in)

        for i in range(N):
            Xi = X[neighbors[i]]
            Xi -= Xi.mean(0)

            # compute n_components largest eigenvalues of Xi * Xi^T
            if use_svd:
                v = svd(Xi, full_matrices=True)[0]
            else:
                Ci = np.dot(Xi, Xi.T)
                v = eigh(Ci)[1][:, ::-1]

            Gi = np.zeros((n_neighbors, n_components + 1))
            Gi[:, 1:] = v[:, :n_components]
            Gi[:, 0] = 1. / np.sqrt(n_neighbors)

            GiGiT = np.dot(Gi, Gi.T)

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] -= GiGiT
            M[neighbors[i], neighbors[i]] += 1
    
    return W
    a,b = null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                      tol=tol, max_iter=max_iter, random_state=random_state)
    
    #return a,b,W





config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


itd = 10

input_bottleneck = Input(shape=(itd,), name="inp_bt")
input_img = Input(shape=(784,), name="inp_img")
input_S = Input(shape=(itd,), name="inp_S")


encoded = Dense(4 * itd, activation='relu')(input_img)
encoded = Dense(2 * itd, activation='relu')(encoded)

bottlenek = Dense(itd, activation='relu', name="bottleneck")(encoded)

decoded = Dense(2 * itd, activation='relu')(bottlenek)
decoded = Dense(4 * itd, activation='relu')(decoded)


output = Dense(784, activation='sigmoid', name="logits")(decoded)


#encoded = Dense(128, activation='relu')(input_img)
#encoded = Dense(64, activation='relu')(encoded)
#bottlenek = Dense(10, activation='relu', name="bottleneck")(encoded)

#decoded = Dense(64, activation='relu')(bottlenek)
#decoded = Dense(128, activation='relu')(bottlenek)
#output = Dense(784, activation='sigmoid', name="logits")(decoded)

final_dense = concatenate(
    [input_bottleneck, 
     output,
     input_S
    ], name="concatenate")

model = Model([input_bottleneck, input_img, input_S], [final_dense, output])

model.load_weights("deepdr_fashion_besst.h5")

bottleneck_model = Model(inputs=model.get_layer("inp_img").input, outputs=model.get_layer("bottleneck").output)

#model = Model(input_img, output)

l = 2
def custom_loss(y_true, y_pred):
    bottle=y_pred[:,:itd]
    pred = y_pred[:,itd:784+itd]
    Sb = y_pred[:,784+itd:]
    
    return tf.norm(y_true - pred) + l*tf.norm(bottle - Sb)


model.compile(optimizer='adam', loss={"concatenate":custom_loss, "logits": None})




sub_train = [x_train[x:x+100] for x in range(0, len(x_train), 100)]

for epoch in range(0,500):
    for x in sub_train:
        
        bottleneck = bottleneck_model.predict(x)
        
        lle = LLEC(n_components=5, n_neighbors=15)
        S = lle.fit_W(bottleneck)
        sdotB = np.dot(S, bottleneck)
        
        lss = model.train_on_batch(
            [
                bottleneck, 
                x,
                sdotB
            ], 
            x
        )
    print("epoch : ", epoch, " : ",lss)
    if epoch % 100 == 0:
        model.save("deepdr_mnist.h5")

    


    


