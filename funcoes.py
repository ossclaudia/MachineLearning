import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import tree
from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------- Função para calcular multicolinearity -------------------

def check_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"Feature {i+1}" for i in range(X.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    print(vif_data)

# ------------------- Função para gerar os data sets -------------------

def generate_datasets():
    datasets = []

    # 1. make_classification simples com 2 classes
    X1, y1 = make_classification(n_samples=500, n_features=2, n_classes=2, random_state=42, n_redundant=0,
                                n_clusters_per_class=1, class_sep=2)
    datasets.append((X1[:, :2], y1, "Base Make Classification"))

    # 2. make_classification com mais instâncias
    X2, y2 = make_classification(n_samples=2000, n_features=2, n_classes=2, random_state=42, n_redundant=0,
                                n_clusters_per_class=1, class_sep=2)
    datasets.append((X2[:, :2], y2, "Make Classification with more instances"))

    # 3. make_classification com classes desbalanceadas (70/30)
    X3, y3 = make_classification(n_samples=500, n_features=2, n_classes=2, random_state=42, n_redundant=0,
                                n_clusters_per_class=1, class_sep=2, weights=[0.7, 0.3])
    datasets.append((X3[:, :2], y3, "Make Classification unbalanced (70/30)"))

    # 4. Make Classification com alto nível de sobreposição 
    X4, y4 = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                                 n_clusters_per_class=1, class_sep=1, random_state=42)
    datasets.append((X4, y4, "Make Classification with high overlap between classes"))

    # 5. make_circles com pouco ruído 
    X5, y5 = make_circles(n_samples=500, noise=0.02, factor=0.8, random_state=42)
    datasets.append((X5, y5, "Make Circles with little noise"))

    # 6. make_circles com mais ruído
    X6, y6 = make_circles(n_samples=500, noise=0.2, factor=0.8, random_state=42)
    datasets.append((X6, y6, "Make Circles with higher noise"))

    # 7. make_blobs gaussian classes and similar covariances
    X7, y7 = make_blobs(n_samples=500, centers=2, n_features=2, random_state=42)
    datasets.append((X7, y7, "Dataset with Gaussian classes & similar covariances"))
    
    # 8. Blobs com dispersão diferente 
    X8, y8 = make_blobs(n_samples=[300, 200], centers=[[0, 0], [5, 5]], 
                         cluster_std=[0.5, 2], random_state=42)
    datasets.append((X8, y8, "Different distribution of points with blobs"))

    return datasets

# ------------------- Funcao da avaliacao dos modelos -------------------

def evaluation(y,y_pred):
    # Scores
    print("classification report: \n", classification_report(y,y_pred))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 
    # Confusion Matrix
    sns.heatmap(confusion_matrix(y, y_pred), 
            annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    auc_score = roc_auc_score(y, y_pred)

    axes[1].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color="blue")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guessing")
    axes[1].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (TPR)")
    axes[1].legend(loc="lower right")
    axes[1].grid()

    plt.tight_layout()
    plt.show()

# ------------------- Funcao de criacao de arvores -------------------

def plottree(model,k):
    if k==1:
        plt.subplots(nrows = 1,ncols = 1,figsize = (7,7), dpi=200)
        tree.plot_tree(model, fontsize=5, filled=True)
        plt.show()
    else: 
        plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
        tree.plot_tree(model, fontsize=5, filled=True)
        plt.show()