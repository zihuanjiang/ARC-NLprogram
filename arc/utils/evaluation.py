# arc/utils/evaluation.py

import numpy as np

def calculate_f1_score(predicted_grid, target_grid):

    # Calculates the F1-score, Precision, and Recall for a predicted ARC grid. The score is 0 if the grid dimensions do not match.
    # Takes in two 2D lists 

    predicted_np = np.array(predicted_grid, dtype=int)
    target_np = np.array(target_grid, dtype=int)

    if predicted_np.shape != target_np.shape:
        return 0.0, 0.0, 0.0

    # A background pixel is assumed to be 0.
    # Create boolean masks for where the grids have color.
    predicted_has_color = predicted_np > 0
    target_has_color = target_np > 0

    # True Positives (TP): Correctly predicted a non-background pixel with the right color.
    correct_color_mask = predicted_np == target_np
    tp_mask = predicted_has_color & target_has_color & correct_color_mask
    tp = np.sum(tp_mask)

    # False Positives (FP): Predicted a color where there should be background.
    fp_mask = predicted_has_color & ~target_has_color
    fp_base = np.sum(fp_mask)

    # False Negatives (FN): Failed to predict a color where one was required.
    fn_mask = ~predicted_has_color & target_has_color
    fn_base = np.sum(fn_mask)
    
    # Color Errors (CE): Predicted a color, but it was the wrong one.
    # This is both a false positive (for the wrong color) and a false negative (for the missing right color).
    color_error_mask = predicted_has_color & target_has_color & ~correct_color_mask
    ce = np.sum(color_error_mask)
    
    fp = fp_base + ce
    fn = fn_base + ce

    # Calculate Precision, Recall, and F1-Score

    # Precision: Of all the pixels we predicted, how many were correct?
    if (tp + fp) == 0:
        # If we predicted nothing, precision is 1.0 if that was correct, else 0.0
        precision = 1.0 if (tp + fn) == 0 else 0.0
    else:
        precision = tp / (tp + fp)

    # Recall: Of all the pixels we should have predicted, how many did we get?
    if (tp + fn) == 0:
        # If there was nothing to predict, recall is 1.0 if we predicted nothing, else 0.0
        recall = 1.0 if (tp + fp) == 0 else 0.0
    else:
        recall = tp / (tp + fn)

    # F1-Score: The harmonic mean of precision and recall.
    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        
    return f1_score, precision, recall
