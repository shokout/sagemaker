from __future__ import print_function

import argparse
import os

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.datasets.samples_generator import make_blobs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--min_samples', type=int, default=10)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the file and read it into a numpy array
    input_file = os.path.join(args.train, 'train_data.csv')
    if len(input_file) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    train_data = np.genfromtxt(input_file, delimiter=',')

    input_file = os.path.join(args.train, 'true_data.csv')
    if len(input_file) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    labels_true = np.genfromtxt(input_file, delimiter=',')

    # Note that you can add as many as your training my require in the ArgumentParser above.
    eps = args.eps
    min_samples = args.min_samples

    # Now use scikit-learn's dbscan classifier to train the model.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(train_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
#    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(train_data, labels))

    # save the coefficients
    joblib.dump(db, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    db = joblib.load(os.path.join(model_dir, "model.joblib"))
    return db

