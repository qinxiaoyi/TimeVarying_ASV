from sklearn.cluster import KMeans
import numpy as np
from scipy.linalg import eig, eigh
from scipy.ndimage import gaussian_filter


#################### Spectral Clustering ####################

def spectral_clustering(mat, n_clusters=None, threshold=0.99, norm=True):
    '''
    mat: 2d-array affinity matrix (N * N).
    n_clusters: the number of clusters to be determined. 
    threshold: If `n_clusters == None`, then we use threshold to determine the number of clusters.
    ----------
    return: clustered labels (N, ).

    --------------------------------------------------
    Notice: 
    Normalization does not get the best results, but makes the threshold stable (0.99).
    For dihard_2019_dev set, the better results can be gained with config:
    `threhsold = 0.85`, `norm=False`
    '''

    if mat.shape == (1,1):
      return np.array([0], dtype=np.int32)

    # Normalization. Reference: `Speaker diarization with LSTM`.
    if norm:
        # mat = gaussian_filter(mat, sigma=1)
        # mat[mat<0.01] = 0
        mat = np.maximum(mat, mat.T)
        mat = np.dot(mat, mat.T)
        mat = mat/(np.max(mat, axis=1) + 1e-15)

    W = (mat + mat.T)/2
    np.fill_diagonal(W, 0)
    D = np.sum(W, axis=0)
    D_half = D ** 0.5 + 1e-15
    # Method 1 of the normalization of the Laplacian matrix
    L = (np.diag(D) - W) / D_half[:,np.newaxis] / D_half
    # # Method 2 of the normalization of the Laplacian matrix
    # L = (np.diag(D) - W)/D[:,np.newaxis]

    eigval, eigvec = eigh(L)
    # eigval, eigvec = eig(L)
    if(n_clusters == None):
        n_clusters = sum(eigval<threshold)

    idx = np.argsort(eigval)[0:n_clusters]
    eigvec = np.real(eigvec[:, idx])
#     np.set_printoptions(precision=3, suppress=True)
#     print(eigvec)
#     print(np.sort(eigval))
    
    clf = KMeans(n_clusters=n_clusters, 
                init="k-means++",
                max_iter=100, 
                random_state=0)
    s = clf.fit(eigvec)
    return s.labels_


if __name__ == '__main__':
    mat = np.ones([5,5], dtype=np.float32)
    mat[3,:] = 0
    mat[:,3] = 0
    mat[3,3] = 1
    print(mat)
    print('*'*10)
    print(spectral_clustering(mat))
