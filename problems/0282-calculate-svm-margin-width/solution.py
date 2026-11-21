import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    """
    Calculate the margin width of a linear SVM classifier.
    
    Parameters:
    w : np.ndarray - weight vector defining the hyperplane
    
    Returns:
    float - the total margin width
    """
    # margin_width = 2/(np.sqrt(np.sum(w**2)))
    margin_width = 2/(np.sqrt(w @ w))
    return margin_width