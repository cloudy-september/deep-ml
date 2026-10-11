import torch

def calculate_auc(y_true, y_scores) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) using PyTorch.
    
    Args:
        y_true: List, array, or torch.Tensor of binary ground truth labels (0 or 1)
        y_scores: List, array, or torch.Tensor of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """

    # before anything, convert any input type to one standard single type
    if hasattr(y_true, "tolist"):
        y_true = y_true.tolist()
    if hasattr(y_scores, "tolist"):
        y_scores = y_scores.tolist()

    # assert y_true and y_scores have same size
    assert len(y_true) == len(y_scores), "size should be the same"

    # tpr/fpr can be connected as dots in one curve
    roc = []

    # for each in y scores, do the cutoff and at each cutoff estimate tpr and fpr 
    for cutoff in y_scores:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y_scores)):
            if y_scores[i] >= cutoff and y_true[i] == 1:
                tp+=1
            elif y_scores[i] >= cutoff and y_true[i] == 0:
                fp+=1
            elif y_scores[i] < cutoff and y_true[i] == 0:
                tn+=1
            elif y_scores[i] < cutoff and y_true[i] == 1:
                fn+=1
            
        tpr = tp / (tp + fn) # y axis
        fpr = 1 - (tn / (fp + tn)) # x axis
        roc.append((fpr,tpr))
    
    # add (0, 0) and (1,1) to roc since you need to cover the whole area under curve
    roc.append((0,0))
    roc.append((1,1))
    
    # sort first so area can be calculated correctly

    # roc = sorted(roc, key=lambda item: (item[0], item[1]))
    # the above is same as below
    roc = sorted(roc)


    # then auc

    # u calculate auc using trapezoidal rule
    total = 0
    for i in range(len(roc)-1):
        interval = (roc[i+1][0] - roc[i][0]) * (roc[i][1]+roc[i+1][1])
        interval /= 2
        total += interval


    return total





