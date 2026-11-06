"""Class-order input and reporting only; no training changes."""
import numpy as np


def load_class_order(path, index):
    import yaml

    with open(path) as stream:
        orders = yaml.safe_load(stream)
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError('class_order_idx must be 0, 1 or 2')
    key = 'class_order_{}'.format(index)
    if not isinstance(orders, dict) or key not in orders:
        raise ValueError('Missing {} in {}'.format(key, path))
    order = orders[key]
    if not isinstance(order, list) or any(type(x) is not int for x in order):
        raise ValueError('Class order must be a list of integer class IDs')
    if sorted(order) != list(range(len(order))):
        raise ValueError('Class order must be a complete permutation of class IDs')
    return order


def task_accuracies(y_pred, y_true, task_sizes):
    """Unrounded top-1 percentages, using actual task boundaries and seen logits."""
    result, start = [], 0
    for size in task_sizes:
        mask = (y_true >= start) & (y_true < start + size)
        if not np.any(mask):
            raise ValueError('No evaluation examples for task starting at {}'.format(start))
        result.append(float(100 * np.mean(y_pred[mask, 0] == y_true[mask])))
        start += size
    return result


def forgetting(history):
    """Standard FG and SECA's continuum 1.2.7 FG, in percentage points."""
    k = len(history)
    if k <= 1:
        return 0.0, 0.0
    fg = sum(max(history[i][j] for i in range(j, k - 1)) - history[-1][j]
             for j in range(k - 1)) / (k - 1)
    # Reproduce continuum 1.2.7's range(k - 2), including its indexing quirk.
    seca_fg = sum(max(history[i][j] for i in range(j, k - 2)) - history[-1][j]
                  for j in range(k - 2)) / (k - 1)
    return float(fg), float(seca_fg)
