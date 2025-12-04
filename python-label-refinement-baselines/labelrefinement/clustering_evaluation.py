import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import entropy as shannon_entropy
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter

if not os.path.exists('clustering_evaluation.py'):
    # If this fails, copy the evaluate_expected_entropy logic directly into this file
    pass

def get_clustering_from_refined_log(refined_log, imprecise_labels, activity_key="concept:name"):
    """Extract clustering information from refined log - ONLY for imprecise events"""
    clustering = []
    seen_labels = []

    print(f"[CLUSTERING_EXTRACT] Processing {len(refined_log)} traces")
    print(f"[CLUSTERING_EXTRACT] Looking for imprecise labels: {imprecise_labels}")

    events_found = 0
    unique_refined_labels = set()

    for trace in refined_log:
        for event in trace:
            activity = event.get(activity_key, "")

            # Check if this activity is imprecise or a refined version of one
            is_imprecise_related = False

            # Direct match with imprecise labels
            if activity in imprecise_labels:
                is_imprecise_related = True
            else:
                # Check if it's a refined version (e.g., "D_0", "D_1" for imprecise label "D")
                for imprecise_label in imprecise_labels:
                    if activity.startswith(f"{imprecise_label}_"):
                        is_imprecise_related = True
                        break

            # Only include imprecise-related events
            if is_imprecise_related:
                unique_refined_labels.add(activity)
                if activity not in seen_labels:
                    seen_labels.append(activity)
                clustering.append(seen_labels.index(activity))
                events_found += 1

    print(f"[CLUSTERING_EXTRACT] Found {events_found} imprecise events")
    print(f"[CLUSTERING_EXTRACT] Unique refined labels: {seen_labels}")
    print(f"[CLUSTERING_EXTRACT] Clustering sample: {clustering[:10]}...")

    return clustering

def get_ground_truth_clustering_from_log(original_log, imprecise_labels, original_labels, activity_key="concept:name"):
    """Extract ground truth clustering from original log - ONLY for imprecise events"""
    clustering = []

    print(f"[GROUND_TRUTH_EXTRACT] Processing {len(original_log)} traces")
    print(f"[GROUND_TRUTH_EXTRACT] Looking for imprecise labels: {imprecise_labels}")
    print(f"[GROUND_TRUTH_EXTRACT] Original labels: {original_labels}")

    events_found = 0
    ground_truth_labels = set()

    for trace in original_log:
        for event in trace:
            current_activity = event.get(activity_key, "")

            # Only include events that have been made imprecise (current activity is imprecise)
            if current_activity in imprecise_labels:
                # Check if OrgLabel exists for ground truth
                if 'OrgLabel' in event:
                    ground_truth_label = event['OrgLabel']
                    try:
                        clustering.append(original_labels.index(ground_truth_label))
                        ground_truth_labels.add(ground_truth_label)
                        events_found += 1
                    except ValueError:
                        # If OrgLabel not in original_labels, use fallback index
                        clustering.append(0)
                        events_found += 1
                else:
                    # For real logs without ground truth, use activity name as cluster ID
                    clustering.append(0)
                    events_found += 1

    print(f"[GROUND_TRUTH_EXTRACT] Found {events_found} imprecise events")
    print(f"[GROUND_TRUTH_EXTRACT] Unique ground truth labels: {sorted(ground_truth_labels)}")
    print(f"[GROUND_TRUTH_EXTRACT] Clustering sample: {clustering[:10]}...")

    if not clustering:  # If no imprecise labels found, return default
        clustering = [0]

    return clustering

def compute_cluster_entropy(labels, base=2, normalized=True):
    """Compute entropy for a list of cluster labels"""
    if not labels:
        return 0.0
    
    label_counts = pd.Series(labels).value_counts()
    total = len(labels)
    probabilities = label_counts / total
    entropy_val = shannon_entropy(probabilities, base=base)
    
    if normalized and len(label_counts) > 1:
        max_entropy = shannon_entropy([1/len(label_counts)] * len(label_counts), base=base)
        entropy_val = entropy_val / max_entropy if max_entropy > 0 else 0.0
    
    return entropy_val

def compute_expected_entropy_clusters_perspective(refined_clustering, ground_truth_clustering, base=2):
    """
    Compute expected entropy from clusters perspective:
    For each cluster, calculate entropy of ground truth labels within it.
    """
    if not refined_clustering or not ground_truth_clustering:
        return 0.0, {}
    
    if len(refined_clustering) != len(ground_truth_clustering):
        print(f"[WARNING] Clustering length mismatch: {len(refined_clustering)} vs {len(ground_truth_clustering)}")
        return 0.0, {}
    
    # Group ground truth labels by refined clusters
    cluster_to_ground_truth = defaultdict(list)
    for refined_cluster, ground_truth_label in zip(refined_clustering, ground_truth_clustering):
        cluster_to_ground_truth[refined_cluster].append(ground_truth_label)
    
    total_events = len(refined_clustering)
    cluster_entropies = []
    cluster_weights = []
    cluster_details = {}
    
    print(f"\n=== Expected Entropy Analysis: Clusters Perspective ===")
    
    for cluster_id, ground_truth_labels in cluster_to_ground_truth.items():
        cluster_size = len(ground_truth_labels)
        cluster_entropy = compute_cluster_entropy(ground_truth_labels, base=base, normalized=False)
        cluster_weight = cluster_size / total_events
        
        cluster_entropies.append(cluster_entropy)
        cluster_weights.append(cluster_weight)
        
        # Store details
        label_counts = Counter(ground_truth_labels)
        cluster_details[cluster_id] = {
            'size': cluster_size,
            'entropy': cluster_entropy,
            'weight': cluster_weight,
            'label_distribution': dict(label_counts)
        }
        
        print(f"Cluster {cluster_id}: {cluster_size} events, entropy = {cluster_entropy:.4f}")
        for label, count in label_counts.items():
            prob = count / cluster_size
            print(f"  {label}: {count} events ({prob:.1%})")
    
    expected_entropy = np.sum(np.array(cluster_entropies) * np.array(cluster_weights))
    print(f"\nExpected Entropy (Clusters Perspective): {expected_entropy:.4f}")
    
    return expected_entropy, cluster_details

def compute_expected_entropy_labels_perspective(refined_clustering, ground_truth_clustering, base=2):
    """
    Compute expected entropy from labels perspective:
    For each ground truth label, calculate entropy of clusters assigned to it.
    """
    if not refined_clustering or not ground_truth_clustering:
        return 0.0, {}
    
    if len(refined_clustering) != len(ground_truth_clustering):
        print(f"[WARNING] Clustering length mismatch: {len(refined_clustering)} vs {len(ground_truth_clustering)}")
        return 0.0, {}
    
    # Group refined clusters by ground truth labels
    label_to_clusters = defaultdict(list)
    for refined_cluster, ground_truth_label in zip(refined_clustering, ground_truth_clustering):
        label_to_clusters[ground_truth_label].append(refined_cluster)
    
    total_events = len(ground_truth_clustering)
    expected_entropy = 0.0
    label_details = {}
    
    print(f"\n=== Expected Entropy Analysis: Labels Perspective ===")
    
    for ground_truth_label, cluster_assignments in label_to_clusters.items():
        label_size = len(cluster_assignments)
        label_entropy = compute_cluster_entropy(cluster_assignments, base=base, normalized=False)
        label_weight = label_size / total_events
        
        expected_entropy += label_weight * label_entropy
        
        # Store details
        cluster_counts = Counter(cluster_assignments)
        label_details[ground_truth_label] = {
            'size': label_size,
            'entropy': label_entropy,
            'weight': label_weight,
            'cluster_distribution': dict(cluster_counts)
        }
        
        print(f"Label {ground_truth_label}: {label_size} events, entropy = {label_entropy:.4f}")
        for cluster, count in cluster_counts.items():
            prob = count / label_size
            print(f"  {cluster}: {count} events ({prob:.1%})")
    
    print(f"\nExpected Entropy (Labels Perspective): {expected_entropy:.4f}")
    
    return expected_entropy, label_details

def evaluate_refined_log_quality(refined_log, original_log, imprecise_labels, original_labels, parameters, is_real_life_log=False):
    """
    Comprehensive evaluation of refined log quality including:
    - Precision and Fitness
    - F-score
    - Expected Entropy (both perspectives)
    - NMI and ARI
    """
    results = {}
    
    print(f"[DEBUG] Starting evaluation...")
    print(f"[DEBUG] Refined log has {len(refined_log) if refined_log else 0} traces")
    
    # Check if refined log is empty
    if not refined_log or len(refined_log) == 0:
        print(f"[DEBUG] Refined log is empty!")
        return {
            'refined_log_precision': 0.0,
            'refined_log_fitness': 0.0,
            'refined_log_fscore': 0.0,
            'expected_entropy_clusters': 0.0,
            'expected_entropy_labels': 0.0,
            'nmi': 0.0,
            'ari': 0.0
        }
    
    # Check activities in refined log
    activities = set()
    total_events = 0
    for trace in refined_log:
        for event in trace:
            activities.add(event.get('concept:name', 'UNKNOWN'))
            total_events += 1
    
    print(f"[DEBUG] Total events: {total_events}")
    print(f"[DEBUG] Unique activities: {activities}")
    print(f"[DEBUG] Number of unique activities: {len(activities)}")
    
    if len(activities) <= 1:
        print(f"[DEBUG] Only one activity type found - cannot compute meaningful metrics")
        return {
            'refined_log_precision': 0.0,
            'refined_log_fitness': 0.0,
            'refined_log_fscore': 0.0,
            'expected_entropy_clusters': 0.0,
            'expected_entropy_labels': 0.0,
            'nmi': 0.0,
            'ari': 0.0
        }
    
    try:
        # 1. PRECISION AND FITNESS EVALUATION
        print(f"[DEBUG] Attempting model discovery...")
        
        # Discover model from refined log
        process_tree = inductive_miner.apply(refined_log)
        print(f"[DEBUG] Process tree discovered: {type(process_tree)}")
        
        # Convert process tree to Petri net
        refined_net, refined_initial_marking, refined_final_marking = pt_converter.apply(process_tree)
        print(f"[DEBUG] Model discovery successful")
        print(f"[DEBUG] Model has {len(refined_net.places)} places and {len(refined_net.transitions)} transitions")
        
        # Calculate precision (use ETCONFORMANCE_TOKEN for consistency with other approaches)
        print(f"[DEBUG] Computing precision...")
        precision_result = precision_evaluator.apply(refined_log, refined_net, refined_initial_marking, refined_final_marking,
                                                     variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        print(f"[DEBUG] Precision result: {precision_result} (type: {type(precision_result)})")
        results['refined_log_precision'] = precision_result
        
        # Calculate fitness (explicitly use TOKEN_BASED for consistency with other approaches)
        print(f"[DEBUG] Computing fitness...")
        fitness_result = fitness_evaluator.apply(refined_log, refined_net, refined_initial_marking, refined_final_marking,
                                                variant=fitness_evaluator.Variants.TOKEN_BASED)
        print(f"[DEBUG] Fitness result: {fitness_result} (type: {type(fitness_result)})")
        
        if isinstance(fitness_result, dict):
            print(f"[DEBUG] Fitness result keys: {fitness_result.keys()}")
            fitness_val = fitness_result.get('log_fitness', fitness_result.get('fitness', 0.0))
            print(f"[DEBUG] Extracted fitness value: {fitness_val}")
        else:
            fitness_val = float(fitness_result)
            print(f"[DEBUG] Converted fitness value: {fitness_val}")
            
        results['refined_log_fitness'] = fitness_val
        
        # Calculate F-score (harmonic mean of precision and fitness)
        precision_val = results['refined_log_precision']
        print(f"[DEBUG] Calculating F-score with precision={precision_val}, fitness={fitness_val}")
        
        if precision_val > 0 and fitness_val > 0:
            fscore = 2 * (precision_val * fitness_val) / (precision_val + fitness_val)
            print(f"[DEBUG] F-score calculated: {fscore}")
            results['refined_log_fscore'] = fscore
        else:
            print(f"[DEBUG] F-score is 0 because precision={precision_val} or fitness={fitness_val} is 0")
            results['refined_log_fscore'] = 0.0
        
        print(f"[EVALUATION] Precision: {precision_val:.4f}, Fitness: {fitness_val:.4f}, F-score: {results['refined_log_fscore']:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to compute precision/fitness: {e}")
        import traceback
        traceback.print_exc()
        results.update({
            'refined_log_precision': 0.0,
            'refined_log_fitness': 0.0,
            'refined_log_fscore': 0.0
        })
    
    try:
        # 2. CLUSTERING QUALITY EVALUATION (Skip for real-life logs)
        if not is_real_life_log:
            print(f"[EVALUATION] Computing clustering quality metrics...")
            
            # Extract clusterings
            refined_clustering = get_clustering_from_refined_log(refined_log, imprecise_labels, parameters.get("ACTIVITY_KEY", "concept:name"))
            ground_truth_clustering = get_ground_truth_clustering_from_log(original_log, imprecise_labels, original_labels, parameters.get("ACTIVITY_KEY", "concept:name"))
            
            if refined_clustering and ground_truth_clustering:
                # Ensure same length
                min_length = min(len(refined_clustering), len(ground_truth_clustering))
                refined_clustering = refined_clustering[:min_length]
                ground_truth_clustering = ground_truth_clustering[:min_length]
                
                if len(refined_clustering) > 0:
                    # Expected Entropy - Clusters Perspective
                    entropy_clusters, cluster_details = compute_expected_entropy_clusters_perspective(
                        refined_clustering, ground_truth_clustering
                    )
                    results['expected_entropy_clusters'] = entropy_clusters
                    
                    # Expected Entropy - Labels Perspective  
                    entropy_labels, label_details = compute_expected_entropy_labels_perspective(
                        refined_clustering, ground_truth_clustering
                    )
                    results['expected_entropy_labels'] = entropy_labels
                    
                    # NMI and ARI
                    nmi = normalized_mutual_info_score(ground_truth_clustering, refined_clustering)
                    ari = adjusted_rand_score(ground_truth_clustering, refined_clustering)
                    
                    results['nmi'] = nmi
                    results['ari'] = ari
                    
                    print(f"[EVALUATION] Expected Entropy (Clusters): {entropy_clusters:.4f}")
                    print(f"[EVALUATION] Expected Entropy (Labels): {entropy_labels:.4f}")
                    print(f"[EVALUATION] NMI: {nmi:.4f}, ARI: {ari:.4f}")
                    
                    # Additional clustering statistics
                    results['clustering_stats'] = {
                        'num_refined_clusters': len(set(refined_clustering)),
                        'num_ground_truth_labels': len(set(ground_truth_clustering)),
                        'total_events_analyzed': len(refined_clustering)
                    }
                else:
                    print(f"[WARNING] No clustering data available")
                    results.update({
                        'expected_entropy_clusters': 0.0,
                        'expected_entropy_labels': 0.0,
                        'nmi': 0.0,
                        'ari': 0.0
                    })
            else:
                print(f"[WARNING] Failed to extract clustering information")
                results.update({
                    'expected_entropy_clusters': 0.0,
                    'expected_entropy_labels': 0.0,
                    'nmi': 0.0,
                    'ari': 0.0
                })
        else:
            print(f"[EVALUATION] Skipping clustering quality metrics for real-life log")
            results.update({
                'expected_entropy_clusters': 0.0,
                'expected_entropy_labels': 0.0,
                'nmi': 0.0,
                'ari': 0.0
            })
    
    except Exception as e:
        print(f"[ERROR] Failed to compute clustering metrics: {e}")
        results.update({
            'expected_entropy_clusters': 0.0,
            'expected_entropy_labels': 0.0,
            'nmi': 0.0,
            'ari': 0.0
        })
    
    return results