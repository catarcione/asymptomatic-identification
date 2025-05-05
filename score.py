import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

def auc_score(infected_nodes, centrality_measures, observed_nodes):
    """
    Calculate the AUC score and ROC curve, given the set of infected nodes,
    the network centrality measure used for ranking, and a list of observed 
    nodes to be excluded from the calculation.

    Parameters:
    - infected_nodes: List of infected nodes
    - centrality_measures: Dictionary where the keys are node indices and
                           the values are their corresponding centrality
                           measures (e.g., degree, betweenness, etc.)
    - observed_nodes : List of node indices to exclude from evaluation
    
    Returns:
    - auc: The Area Under the Curve (AUC) score
    - curve: Tuple containing the False Positive Rate (FPR),
             True Positive Rate (TPR), and thresholds for
             plotting the ROC curve
    """
    # Filter out observed nodes
    evaluation_nodes = [node for node in centrality_measures if node not in observed_nodes]

    # Build ground truth and scores
    y_true = []
    y_scores = []

    for node in evaluation_nodes:
        y_true.append(1 if node in infected_nodes else 0)
        y_scores.append(centrality_measures[node])

    # Compute the AUC score
    auc = roc_auc_score(y_true, y_scores)
    # Compute the ROC curve
    curve = roc_curve(y_true, y_scores)

    return auc, curve


def top_k_score(infected_nodes, centrality_measures, observed_nodes, top_k):
    """
    Evaluate how well a given centrality measure identifies infected nodes by
    computing precision and recall within the top-k ranked nodes (excluding observed nodes).

    Parameters:
    - infected_nodes: List of infected nodes
    - centrality_measures: Dictionary where the keys are node indices and
                           the values are their corresponding centrality
                           measures (e.g., degree, betweenness, etc.)
    - observed_nodes : List of node indices to exclude from evaluation
    - top_k: The fraction of the unobserved nodes to include in the top-k set
    
    Returns:
    - precision: Precision of predicting infected nodes
                 among the top-k centrality-ranked nodes
    - recall: Recall of predicting infected nodes 
              among the top-k centrality-ranked nodes
    """
    # Filter out observed nodes
    evaluation_nodes = [node for node in centrality_measures if node not in observed_nodes]

    # Build ground truth and scores
    y_true = []
    y_scores = []

    for node in evaluation_nodes:
        y_true.append(1 if node in infected_nodes else 0)
        y_scores.append(centrality_measures[node])

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    top_indices = np.argsort(y_scores)[-top_k:]
    y_true_top = y_true[top_indices]
    y_pred_top = np.ones_like(y_true_top)
    
    precision = precision_score(y_true_top, y_pred_top)
    recall = recall_score(y_true, np.isin(np.arange(len(y_true)), top_indices).astype(int))

    return precision, recall
