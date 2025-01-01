# HomonymMend
Detection and repair of homonymous activity labels in process event streams for online process mining.

1. dynamic_binning_and_categorization.py - Handles Step 1 - Discretization.
2. feature_selection_with_drift_detection.py - Handles Step 2 - Online Feature Selection and Importance Analysis.
3. dynamic_feature_vector_construction.py - Handles Step 3 - Dynamic Feature Vector Construction
4. homonym_detection.py - Handles Step 4 - Homonym Detection (Splitting/Merging, Incremental Clustering, Contextual Weighted Similarity)