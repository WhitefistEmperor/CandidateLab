"""Exact decision-path decomposition of a forest's selected-class model score."""
import numpy as np

def forest_contributions(pipeline, row):
    inputs = pipeline.named_steps['impute'].transform(row) if 'impute' in pipeline.named_steps else row
    # sklearn trees cast prediction inputs to float32 before comparing thresholds.
    x = pipeline.named_steps['scale'].transform(inputs)[0].astype(np.float32)
    forest = pipeline.named_steps['model']
    selected_index = list(forest.classes_).index(1)
    contributions = np.zeros(len(x))
    baseline = 0.0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        node = 0
        def score(index):
            distribution = tree.value[index][0]
            return distribution[selected_index] / distribution.sum()
        baseline += score(node)
        while tree.children_left[node] != tree.children_right[node]:
            feature = tree.feature[node]
            next_node = tree.children_left[node] if x[feature] <= tree.threshold[node] else tree.children_right[node]
            contributions[feature] += score(next_node) - score(node)
            node = next_node
    return baseline / len(forest.estimators_), contributions / len(forest.estimators_)
