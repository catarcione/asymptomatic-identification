import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def auc_score(infected_nodes, centrality_measures):
    """
    Calculate the AUC score and ROC curve, given the set of infected nodes
    and the network centrality measure used for ranking.

    Parameters:
    - infected_nodes: List of infected nodes
    - centrality_measures: Dictionary where the keys are node indices and
                           the values are their corresponding centrality
                           measures (e.g., degree, betweenness, etc.)
    
    Returns:
    - auc: The Area Under the Curve (AUC) score
    - curve: Tuple containing the False Positive Rate (FPR),
             True Positive Rate (TPR), and thresholds for
             plotting the ROC curve
    """
    centrality_values = list(centrality_measures.values())
    n = len(centrality_values)

    # Ground truth labels
    y_true = np.zeros(n)
    # Set the infected nodes to have a label of 1
    for i in infected_nodes:
        y_true[i] = 1

    # Compute the AUC score
    auc = roc_auc_score(y_true, centrality_values)
    # Compute the ROC curve
    curve = roc_curve(y_true, centrality_values)

    return auc, curve
