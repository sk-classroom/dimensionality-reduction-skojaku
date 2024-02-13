# %%
import unittest
import numpy as np
import sys
import pandas as pd

sys.path.append("assignments/")
from assignment import *
from scipy import stats
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class TestDimensionalityReduction(unittest.TestCase):
    def setUp(self):

        self.X, self.y = make_blobs(
            n_samples=100, centers=3, n_features=3, random_state=42
        )

    def test_pca_fit_transform(self):
        pca = PrincipalComponentAnalysis(n_components=2)
        pca.fit(self.X)
        X_transformed = pca.transform(self.X)

        explained_variance = np.trace(np.cov(X_transformed.T)) / np.trace(
            np.cov(self.X.T)
        )
        assert (
            explained_variance > 0.9
        ), "the explained variance should be greater than 0.9"
        self.assertEqual(X_transformed.shape, (100, 2))

        assert np.all(
            X_transformed.mean(axis=0) < 1e-5
        ), "The projected data must have zero mean. It's likely that you forgot to center the data before projection"
        self.assertEqual(X_transformed.shape, (100, 2))

    def test_lda_fit_transform(self):
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(self.X, self.y)
        X_transformed = lda.transform(self.X)

        model = SVC().fit(X_transformed, self.y)
        score = model.score(X_transformed, self.y)
        assert score > 0.9, "the class must be clealry separated"
        assert np.all(
            X_transformed.mean(axis=0) < 1e-5
        ), "The projected data must have zero mean. It's likely that you forgot to center the data before projection"
        self.assertEqual(X_transformed.shape, (100, 2))


class TestAdversarialExample(unittest.TestCase):
    def setUp(self):
        self.n_samples = 1000
        self.n_features = 3

    def test_pca_adversarial_example(self):

        dataset = AdversarialExamples()

        min_score = 0.5

        for i in range(10):
            X, y = dataset.pca_adversarial_data(
                n_samples=self.n_samples, n_features=self.n_features
            )
            pca = PCA(n_components=1)
            X_transformed = pca.fit_transform(X)

            k = np.max(y + 1)
            model = KMeans(n_clusters=int(k))
            model.fit(X)
            y_pred = model.predict(X)

            model = KMeans(n_clusters=int(k))
            model.fit(X_transformed)
            y_pred_transformed = model.predict(X_transformed)

            score = normalized_mutual_info_score(y, y_pred)
            score_transformed = normalized_mutual_info_score(y, y_pred_transformed)

            if score > 0.99 and score_transformed < min_score:
                break

        assert (
            score > 0.99
        ), f"The generated data does not have well-separated clusters. separability = (original: {score}, pca: {score_transformed})"

        assert (
            score_transformed < min_score
        ), f"The clusters should not be separable in the PCA projection. separability = (original: {score}, pca: {score_transformed})"


if __name__ == "__main__":
    unittest.main()

# %%
