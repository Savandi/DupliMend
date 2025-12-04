# Baseline Evaluation Methodology

This document explains the evaluation methodology used by both baseline algorithms (Label Refinement and PM-Label-Splitting) when assessing refined process models against imprecise event logs.

---

## Overview

Both baseline algorithms follow a three-step evaluation approach:

1. **Model Discovery**: Process model is discovered from the refined/split event log
2. **Label Relabeling**: Model transitions are relabeled back to original imprecise label space
3. **Conformance Checking**: Relabeled model is evaluated against the original imprecise log

This methodology ensures that the refined model structure (improved control-flow) is evaluated fairly against the original data, measuring how well the refinement improves conformance metrics.

---

## Label Refinement Baseline

### Implementation Location
`labelrefinement/precision_util.py` lines 37-54

### Methodology

**Step 1: Model Discovery** (lines 30-32)
- Input: Refined log containing refined labels (e.g., `D_X_1`, `D_X_2`, `E_X_1`)
- Process: Inductive Miner discovers a Petri net from the refined log
- Output: Process model with refined label names on transitions

**Step 2: Label Relabeling** (lines 37-42)
```python
# Map refined labels back to their original imprecise labels
for transition in net.transitions:
    if transition.label != None and "_X_" in transition.label:
        base_label = transition.label.split("_X_", 1)[0]
        if base_label in imprecise_labels:
            transition.label = base_label
```

This step transforms refined labels back to their original imprecise form:
- `D_X_1` → `D`
- `D_X_2` → `D`
- `E_X_1` → `E`

The pattern matching uses the `_X_` separator to identify and extract the base label name.

**Step 3: Conformance Evaluation** (lines 51-54)
```python
prec = precision_evaluator.apply(event_log, net, initial_marking, final_marking,
                                   variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
simplicity = simplicity_evaluator.apply(net)
```

The relabeled model is replayed on the original imprecise log using PM4Py's conformance checking algorithms.

### Rationale

The relabeling step is critical because:
- The refined model has better control-flow structure (distinguishes different contexts)
- The model must use original label names to replay on the original imprecise log
- Metrics measure: "How much does the refined structure improve conformance over imprecise data?"

Without relabeling, the model could not be evaluated against the original log, as label names would not match.

---

## PM-Label-Splitting Baseline

### Implementation Location
`pm-label-splitting/evaluation/apply_im.py` and `pm-label-splitting/pipeline/post_processor.py`

### Methodology

**Step 1: Model Discovery** (apply_im.py lines 45-46)
- Input: Split log containing split labels (e.g., `D_0`, `D_1`, `E_0`)
- Process: Inductive Miner discovers a Petri net from the split log
- Output: Process model with split label names on transitions

**Step 2: Label Relabeling** (post_processor.py lines 18-28)
```python
def post_process_petri_net(self, net: PetriNet) -> PetriNet:
    """Renames split labels from X_1, X_2,... to X to enable evaluation"""
    for transition in net.transitions:
        if transition.label is not None and transition.label[0] in self._split_labels_to_original_labels.values():
            transition.label = transition.label[0]
    return net
```

The `PostProcessor` class maps split labels back to original forms:
- `D_0` → `D`
- `D_1` → `D`
- `E_0` → `E`

**Step 3: Conformance Evaluation** (apply_im.py lines 50-56)
```python
performance_evaluator = PerformanceEvaluator(final_net,
                                             initial_marking,
                                             final_marking,
                                             original_log,
                                             outfile,
                                             skip_fitness=True)
performance_evaluator.evaluate_performance()
```

The relabeled model is evaluated against the original imprecise log using the same PM4Py conformance checking algorithms.

### Rationale

Similar to Label Refinement, the post-processing step ensures:
- The improved model structure from label splitting is preserved
- Label names match those in the original imprecise log
- Conformance metrics reflect the improvement from splitting imprecise labels

---

## Metrics Calculated

### Real-World Logs (No Ground Truth)

When ground truth is unavailable, only process mining conformance metrics are calculated:

- **Precision**: Measures how much extra behavior the model allows beyond what is observed in the log
- **Fitness**: Measures how well the model can replay traces from the log
- **Simplicity**: Measures model complexity (fewer nodes/arcs is better)
- **Generalization**: Measures how well the model generalizes beyond the training data
- **F-score**: Harmonic mean of precision and fitness

The following metrics **cannot** be calculated without ground truth:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Expected Entropy (Clusters Perspective)
- Expected Entropy (Labels Perspective)

### Synthetic Logs (With Ground Truth)

When ground truth is available (e.g., synthetic datasets with known true labels), all conformance metrics above are calculated **plus** clustering quality metrics:

- **ARI**: Measures similarity between predicted and ground truth clusterings
- **NMI**: Measures mutual information between predicted and ground truth
- **Expected Entropy (Clusters)**: Entropy of ground truth labels within each predicted cluster
- **Expected Entropy (Labels)**: Entropy of predicted clusters within each ground truth label

---

## Implementation Details

### Label Refinement
- Location: `labelrefinement/precision_util.py`
- Pattern: String splitting on `_X_` separator
- Mapping: Direct replacement of transition labels in Petri net

### PM-Label-Splitting
- Location: `pm-label-splitting/pipeline/post_processor.py`
- Pattern: Dictionary-based mapping from split to original labels
- Mapping: Transition label replacement via `PostProcessor` class

Both implementations achieve the same result: a process model with refined structure but original label names, enabling fair comparison against the imprecise log.

---

## Verification

To inspect the relabeling implementation:

**Label Refinement:**
```bash
cat python-label-refinement-baselines/labelrefinement/precision_util.py
# See lines 37-42 for relabeling logic
```

**PM-Label-Splitting:**
```bash
cat python-label-refinement-baselines/pm-label-splitting/pipeline/post_processor.py
# See lines 18-28 for post_process_petri_net() method
```

---

## Summary

Both baseline algorithms implement the same conceptually correct evaluation methodology:

1. Refined/split models are discovered from refined/split logs (better structure)
2. Model labels are mapped back to original imprecise label space (enable evaluation)
3. Conformance metrics are calculated by replaying on original imprecise log (fair comparison)

This approach measures the improvement in model quality attributable to label refinement/splitting, while maintaining compatibility with the original imprecise event log.
