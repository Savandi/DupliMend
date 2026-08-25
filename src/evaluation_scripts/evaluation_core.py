import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy as shannon_entropy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import evaluation_config

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog, Trace, Event

single_eval_config = evaluation_config.get("single_evaluation_config", {})

tracking_dir = os.environ.get('TRACKING_DIR', single_eval_config.get("default_tracking_dir", ""))
event_vector_path = os.environ.get('EVENT_VECTOR_PATH', f"{tracking_dir}/event_feature_vectors.jsonl" if tracking_dir else "")
centroid_path = os.environ.get('CENTROID_PATH', f"{tracking_dir}/final_centroids.json" if tracking_dir else "")
ground_truth_path = os.environ.get('GROUND_TRUTH_PATH', single_eval_config.get("default_ground_truth_path", ""))
activity_of_interest = os.environ.get('ACTIVITY', single_eval_config.get("default_activity", "A"))
event_id_column = os.environ.get('EVENT_ID_COLUMN', single_eval_config.get("event_id_column", "EventID"))
case_id_column = os.environ.get('CASE_ID_COLUMN', single_eval_config.get("case_id_column", "CaseID"))
control_flow_column = os.environ.get('CONTROL_FLOW_COLUMN', single_eval_config.get("control_flow_column", "base_activity_label"))
control_flow_column_ground_truth = os.environ.get('GROUND_TRUTH_COLUMN', single_eval_config.get("control_flow_column_ground_truth", "ground_truth_activity_label"))
test_output_dir = os.environ.get('OUTPUT_DIR', single_eval_config.get("default_output_dir", ""))

inductive_miner_noise_threshold = float(os.environ.get('NOISE_THRESHOLD', single_eval_config.get("inductive_miner_noise_threshold", 0.2)))

full_log_evaluation = os.environ.get('FULL_LOG_EVALUATION', '0').lower() not in ('0', 'false', 'no')


def load_event_vectors(event_vector_path):
    events = []
    try:
        with open(event_vector_path, 'r') as f:
            for line in f:
                event_data = json.loads(line.strip())
                events.append(event_data)
    except Exception as e:
        print(f"Error loading event vectors: {e}")
        return []

    print(f"Loaded {len(events)} event vectors")
    return events


def load_centroids(centroid_path):
    try:
        with open(centroid_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading centroids: {e}")
        return {}

    if 'centroids_by_activity' in data:
        centroids_by_activity = data['centroids_by_activity']

        flattened_centroids = {}
        for activity, clusters in centroids_by_activity.items():
            for cluster_info in clusters:
                cluster_id = cluster_info['cluster_id']
                unique_key = f"{activity}_{cluster_id}"
                flattened_centroids[unique_key] = {
                    'centroid': cluster_info['centroid'],
                    'activity': activity,
                    'original_cluster_id': cluster_id,
                    'samples': cluster_info.get('samples', 0)
                }

        print(f"Loaded centroids: {len(flattened_centroids)} total clusters across {len(centroids_by_activity)} activities")
        print(f"Activities found: {list(centroids_by_activity.keys())}")
        return flattened_centroids
    else:
        print(f"Loaded centroids (legacy format): {len(data)} clusters")
        return data


def load_ground_truth(ground_truth_path):
    try:
        gt_df = pd.read_csv(ground_truth_path)
        gt_df[event_id_column] = gt_df[event_id_column].astype(str)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return pd.DataFrame()

    print(f"Loaded ground truth with {len(gt_df)} events")
    return gt_df


def compute_cluster_entropy(labels, base=2):
    if not labels:
        return 0

    label_counts = pd.Series(labels).value_counts()
    probabilities = label_counts / len(labels)

    if base == 2:
        entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
    else:
        entropy = -sum(probabilities * np.log(probabilities + 1e-10))

    return entropy


def shannon_entropy(probabilities, base=2):
    if base == 2:
        return -sum(probabilities * np.log2(probabilities + 1e-10))
    else:
        return -sum(probabilities * np.log(probabilities + 1e-10))


def assign_events_to_clusters(events, centroids):
    from sklearn.metrics.pairwise import cosine_distances

    assigned_events = []

    if activity_of_interest != "ALL":
        filtered_centroids = {k: v for k, v in centroids.items() if v.get('activity') == activity_of_interest}
        print(f"Filtering centroids for activity '{activity_of_interest}': {len(filtered_centroids)} clusters")
    else:
        filtered_centroids = centroids
        print(f"Using all centroids: {len(filtered_centroids)} clusters")

    if not filtered_centroids:
        print("No centroids available after filtering")
        return []

    events_with_embeddings = 0
    events_without_embeddings = 0

    for event in events:
        event_id = str(event.get('event_id', ''))
        embedding = np.array(event.get('feature_vector', []))

        if len(embedding) == 0:
            events_without_embeddings += 1
            if events_without_embeddings <= 3:
                print(f"  DEBUG: Event {events_without_embeddings} fields: {list(event.keys())}")
            continue
        else:
            events_with_embeddings += 1

        min_distance = float('inf')
        nearest_cluster = None
        nearest_activity = None

        for cluster_key, cluster_data in filtered_centroids.items():
            if 'centroid' not in cluster_data:
                continue

            centroid = np.array(cluster_data['centroid'])

            if len(centroid) == len(embedding):
                distance = cosine_distances([embedding], [centroid])[0][0]
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = cluster_key
                    nearest_activity = cluster_data.get('activity', 'unknown')

        if nearest_cluster is not None:
            assigned_events.append({
                'event_id': event_id,
                'nearest_cid': nearest_cluster,
                'nearest_activity': nearest_activity,
                'distance': min_distance
            })

    print(f"Event assignment summary:")
    print(f"  - Events with embeddings: {events_with_embeddings}")
    print(f"  - Events without embeddings: {events_without_embeddings}")
    print(f"  - Events assigned to clusters: {len(assigned_events)}")
    return assigned_events


def merge_with_ground_truth(assigned_events, gt_df):
    assigned_df = pd.DataFrame(assigned_events)
    if assigned_df.empty:
        print("No assigned events to merge")
        return []

    assigned_df['event_id'] = assigned_df['event_id'].astype(str)

    has_base_activity = control_flow_column in gt_df.columns

    merge_columns = [event_id_column, control_flow_column_ground_truth]
    if has_base_activity and activity_of_interest != "ALL":
        merge_columns.append(control_flow_column)

    merged = pd.merge(
        assigned_df,
        gt_df[merge_columns],
        left_on='event_id',
        right_on=event_id_column,
        how='inner'
    )

    print(f"Merged {len(merged)} events with ground truth (from {len(assigned_events)} assigned)")

    if activity_of_interest != "ALL":
        before_filter = len(merged)
        if has_base_activity:
            merged = merged[merged[control_flow_column] == activity_of_interest]
            print(f"Activity filter '{activity_of_interest}': {len(merged)} events (from {before_filter})")
        else:
            print(f"Warning: Column '{control_flow_column}' not found, skipping activity filter")

    if merged.empty:
        print("No events remaining after activity filtering")
        return []

    reassigned = []
    for _, row in merged.iterrows():
        reassigned.append({
            'event_id': row['event_id'],
            'nearest_cid': row['nearest_cid'],
            'nearest_activity': row.get('nearest_activity', 'unknown'),
            'ground_truth': row[control_flow_column_ground_truth],
            'distance': row['distance']
        })

    return reassigned


def calculate_expected_entropy(reassigned):
    cluster_to_labels = {}
    for r in reassigned:
        cid = r['nearest_cid']
        if cid not in cluster_to_labels:
            cluster_to_labels[cid] = []
        cluster_to_labels[cid].append(r['ground_truth'])

    total_events = len(reassigned)
    expected_entropy_clusters = 0

    label_distribution = {}
    for cid, labels in cluster_to_labels.items():
        cluster_size = len(labels)
        entropy = compute_cluster_entropy(labels, base=2)
        weight = cluster_size / total_events
        expected_entropy_clusters += weight * entropy
        label_distribution[cid] = labels

    label_to_clusters = {}
    for r in reassigned:
        gt_label = r['ground_truth']
        if gt_label != "unknown":
            if gt_label not in label_to_clusters:
                label_to_clusters[gt_label] = []
            label_to_clusters[gt_label].append(r['nearest_cid'])

    expected_entropy_labels = 0
    total_labeled_events = sum(len(clusters) for clusters in label_to_clusters.values())

    for gt_label, cluster_list in label_to_clusters.items():
        label_size = len(cluster_list)
        cluster_counts = pd.Series(cluster_list).value_counts()
        probabilities = cluster_counts / len(cluster_list)
        entropy = shannon_entropy(probabilities, base=2)
        weight = label_size / total_labeled_events
        expected_entropy_labels += weight * entropy

    return expected_entropy_clusters, expected_entropy_labels, label_distribution


def calculate_clustering_metrics(reassigned, feature_vectors=None):
    if not reassigned:
        return {}

    true_labels = [r['ground_truth'] for r in reassigned if r['ground_truth'] != "unknown"]
    pred_labels = [r['nearest_cid'] for r in reassigned if r['ground_truth'] != "unknown"]
    event_ids = [r['event_id'] for r in reassigned if r['ground_truth'] != "unknown"]

    if not true_labels:
        return {}

    print(f"Calculating metrics for {len(true_labels)} events")
    print(f"Unique ground truth labels: {len(set(true_labels))}")
    print(f"Unique predicted clusters: {len(set(pred_labels))}")

    try:
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        metrics = {
            'nmi': nmi,
            'ari': ari
        }

        if feature_vectors is not None and len(set(pred_labels)) > 1:
            try:
                X = []
                labels_for_silhouette = []
                for event_id, pred_label in zip(event_ids, pred_labels):
                    if event_id in feature_vectors:
                        X.append(feature_vectors[event_id])
                        labels_for_silhouette.append(pred_label)

                if len(X) > 1 and len(set(labels_for_silhouette)) > 1:
                    X = np.array(X)
                    silhouette = silhouette_score(X, labels_for_silhouette, metric='cosine')
                    metrics['silhouette'] = silhouette
                    print(f"Silhouette score: {silhouette:.6f}")
                else:
                    print("Not enough samples or clusters for Silhouette score")
                    metrics['silhouette'] = None
            except Exception as e:
                print(f"Error calculating Silhouette score: {e}")
                metrics['silhouette'] = None
        else:
            if feature_vectors is None:
                print("Feature vectors not provided - skipping Silhouette score")
            elif len(set(pred_labels)) <= 1:
                print("Only one cluster found - Silhouette score requires at least 2 clusters")
            metrics['silhouette'] = None

        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {}


def main():
    print(f"Starting evaluation for activity: {activity_of_interest}")
    print(f"Event vectors: {event_vector_path}")
    print(f"Centroids: {centroid_path}")
    print(f"Ground truth: {ground_truth_path}")

    events = load_event_vectors(event_vector_path)
    if not events:
        print("No events loaded. Exiting.")
        return

    centroids = load_centroids(centroid_path)
    if not centroids:
        print("No centroids loaded. Exiting.")
        return

    gt_df = load_ground_truth(ground_truth_path)
    if gt_df.empty:
        print("No ground truth loaded. Exiting.")
        return

    print("Assigning events to clusters...")
    assigned_events = assign_events_to_clusters(events, centroids)

    if not assigned_events:
        print("No events were assigned to clusters. Exiting.")
        return

    print("Merging with ground truth...")
    reassigned = merge_with_ground_truth(assigned_events, gt_df)

    if not reassigned:
        print("No events matched with ground truth. Exiting.")
        return

    print(f"Successfully matched {len(reassigned)} events")

    print("Calculating expected entropy...")
    expected_entropy_clusters, expected_entropy_labels, label_distribution = calculate_expected_entropy(reassigned)

    print("Calculating clustering metrics...")
    metrics = calculate_clustering_metrics(reassigned)

    print("\n=== Computing Process Mining Metrics ===")
    print("Comparing: Mlab vs Mre = β(D(Lre))")

    result = create_pm4py_log_from_reassigned_events(reassigned, gt_df, create_imprecise_version=True)
    if result is not None and len(result) == 3:
        refined_log, imprecise_log, label_mapping = result

        print("\n[1/2] Computing Mlab = D(Llab)...")
        imprecise_metrics = compute_process_mining_metrics(imprecise_log)

        print("\n[2/2] Computing Mre = β(D(Lre))...")
        refined_metrics = compute_process_mining_metrics_with_label_mapping(
            refined_log, imprecise_log, label_mapping
        )
    else:
        imprecise_metrics = None
        refined_metrics = None

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")

    print(f"\n--- Clustering Quality Metrics ---")
    print(f"Expected Entropy (Clusters Perspective): {expected_entropy_clusters:.6f}")
    print(f"Expected Entropy (Labels Perspective): {expected_entropy_labels:.6f}")

    if metrics:
        print(f"Normalized Mutual Information (NMI): {metrics['nmi']:.6f}")
        print(f"Adjusted Rand Index (ARI): {metrics['ari']:.6f}")

    print("\n--- Process Mining Quality Metrics ---")

    if imprecise_metrics:
        print("\nMlab = D(Llab):")
        print(f"  Log Precision: {imprecise_metrics['precision']:.6f}")
        print(f"  Log Fitness:   {imprecise_metrics['fitness']:.6f}")
        print(f"  F-Score:       {imprecise_metrics['fscore']:.6f}")
        print(f"  Generalization: {imprecise_metrics['generalization']:.6f}")
        print(f"  Simplicity:     {imprecise_metrics['simplicity']:.6f}")

    if refined_metrics:
        print("\nMre = β(D(Lre)):")
        print(f"  Log Precision: {refined_metrics['precision']:.6f}")
        print(f"  Log Fitness:   {refined_metrics['fitness']:.6f}")
        print(f"  F-Score:       {refined_metrics['fscore']:.6f}")
        print(f"  Generalization: {refined_metrics['generalization']:.6f}")
        print(f"  Simplicity:     {refined_metrics['simplicity']:.6f}")

        if imprecise_metrics:
            precision_improvement = refined_metrics['precision'] - imprecise_metrics['precision']
            fitness_improvement = refined_metrics['fitness'] - imprecise_metrics['fitness']
            fscore_improvement = refined_metrics['fscore'] - imprecise_metrics['fscore']

            print(f"\n{'─'*70}")
            print("IMPROVEMENT OVER BASELINE:")
            print(f"{'─'*70}")
            print(f"  Δ Log Precision: {precision_improvement:+.6f}  {'✓' if precision_improvement >= 0 else '✗'}")
            print(f"  Δ Log Fitness:   {fitness_improvement:+.6f}  {'✓' if fitness_improvement >= 0 else '✗'}")
            print(f"  Δ F-Score:       {fscore_improvement:+.6f}  {'✓' if fscore_improvement >= 0 else '✗'}")

            success_count = sum([
                precision_improvement >= 0,
                fitness_improvement >= 0
            ])
            print(f"\nGoal Achievement: {success_count}/2 criteria met")
            if success_count == 2:
                print("Status: ✓ Goal SATISFIED")

    print(f"Total events evaluated: {len(reassigned)}")
    print(f"Unique clusters: {len(set(r['nearest_cid'] for r in reassigned))}")
    print(f"Unique ground truth labels: {len(set(r['ground_truth'] for r in reassigned if r['ground_truth'] != 'unknown'))}")


def _event_sort_key(series):
    """Order events numerically when the ids are numeric, else lexicographically."""
    numeric = pd.to_numeric(series, errors='coerce')
    return numeric if numeric.notna().all() else series


def create_pm4py_log_from_reassigned_events(reassigned, gt_df, create_imprecise_version=False):
    try:
        if not reassigned:
            print("No reassigned events to create PM4Py log")
            return None

        reassigned_df = pd.DataFrame(reassigned)

        gt_df_clean = gt_df.copy()
        gt_df_clean[event_id_column] = gt_df_clean[event_id_column].astype(str)
        reassigned_df['event_id'] = reassigned_df['event_id'].astype(str)

        if full_log_evaluation:
            merged = pd.merge(
                gt_df_clean,
                reassigned_df,
                left_on=event_id_column,
                right_on='event_id',
                how='left',
                suffixes=('', '_reassigned')
            )
            merged['event_id'] = merged[event_id_column]

            if activity_of_interest and control_flow_column in merged.columns:
                unrefined_target = (
                    merged['nearest_cid'].isna()
                    & (merged[control_flow_column] == activity_of_interest)
                )
                dropped = int(unrefined_target.sum())
                if dropped:
                    print(f"Excluding {dropped} warm-up '{activity_of_interest}' "
                          f"events with no cluster assignment")
                    merged = merged[~unrefined_target]
        else:
            merged = pd.merge(
                reassigned_df,
                gt_df_clean,
                left_on='event_id',
                right_on=event_id_column,
                how='inner'
            )

        if merged.empty:
            print("No events could be merged with ground truth for PM4Py log creation")
            return None

        if case_id_column in merged.columns:
            case_column = case_id_column
        else:
            print(f"Case column '{case_id_column}' not found, using event_id prefix as case")
            merged['extracted_case'] = merged['event_id'].str.extract(r'([^_]+_[^_]+)')
            case_column = 'extracted_case'

        event_log = EventLog()
        imprecise_log = EventLog() if create_imprecise_version else None
        label_mapping = {} if create_imprecise_version else None

        print(f"Creating PM4Py log from {len(merged)} events...")

        for case_id, case_group in merged.groupby(case_column):
            if pd.isna(case_id):
                continue

            trace = Trace()
            trace.attributes['case:concept:name'] = str(case_id)

            imprecise_trace = Trace() if create_imprecise_version else None
            if imprecise_trace is not None:
                imprecise_trace.attributes['case:concept:name'] = str(case_id)

            case_group_sorted = case_group.sort_values(
                'event_id', key=_event_sort_key)

            for _, row in case_group_sorted.iterrows():
                event = Event()

                original_label = row.get(control_flow_column)
                cluster_id = row.get('nearest_cid')
                was_refined = pd.notnull(cluster_id)

                if was_refined:
                    base_activity = row.get('nearest_activity', 'unknown')
                    refined_activity = f"{base_activity}_{cluster_id}"
                else:
                    refined_activity = (original_label
                                        if pd.notnull(original_label) else 'unknown')

                event['concept:name'] = refined_activity

                event['EventID'] = row['event_id']
                event['cluster_id'] = cluster_id if was_refined else None
                event['ground_truth'] = row.get('ground_truth')
                event['distance'] = row.get('distance')

                if 'SYSCALL_timestamp' in row and pd.notnull(row['SYSCALL_timestamp']):
                    try:
                        event['time:timestamp'] = pd.to_datetime(row['SYSCALL_timestamp'], unit='s')
                    except:
                        pass

                trace.append(event)

                if create_imprecise_version and imprecise_trace is not None:
                    imprecise_event = Event()

                    if pd.notnull(original_label):
                        imprecise_label = original_label
                    else:
                        gt = row.get('ground_truth')
                        gt = gt if pd.notnull(gt) and gt != 'unknown' else refined_activity
                        imprecise_label = gt.split('_')[0] if '_' in str(gt) else gt

                    if refined_activity not in label_mapping:
                        label_mapping[refined_activity] = imprecise_label

                    imprecise_event['concept:name'] = imprecise_label
                    imprecise_event['OrgLabel'] = refined_activity
                    imprecise_event['EventID'] = row['event_id']
                    imprecise_event['cluster_id'] = cluster_id if was_refined else None
                    imprecise_event['ground_truth'] = row.get('ground_truth')
                    imprecise_event['distance'] = row.get('distance')

                    if 'SYSCALL_timestamp' in row and pd.notnull(row['SYSCALL_timestamp']):
                        try:
                            imprecise_event['time:timestamp'] = pd.to_datetime(row['SYSCALL_timestamp'], unit='s')
                        except:
                            pass

                    imprecise_trace.append(imprecise_event)

            if len(trace) > 0:
                event_log.append(trace)
                if create_imprecise_version and imprecise_trace is not None and len(imprecise_trace) > 0:
                    imprecise_log.append(imprecise_trace)

        total_events = sum(len(trace) for trace in event_log)
        print(f"Created PM4Py log with {len(event_log)} traces and {total_events} events")

        if create_imprecise_version:
            unique_imprecise = len(set(label_mapping.values()))
            print(f"Label mapping: {len(label_mapping)} refined labels → {unique_imprecise} imprecise labels")
            return event_log, imprecise_log, label_mapping
        else:
            return event_log

    except Exception as e:
        print(f"Error creating PM4Py log: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_process_mining_metrics_with_label_mapping(refined_log, imprecise_log=None, label_mapping=None):
    if refined_log is None or len(refined_log) == 0:
        print("Error: Empty or invalid refined log")
        return None

    if imprecise_log is not None and label_mapping is not None:
        print("\n=== Process Mining with Label Mapping ===")
        print(f"Discovering model from refined log ({len(refined_log)} traces)...")
        print(f"Will evaluate on imprecise log ({len(imprecise_log)} traces)...")

        try:
            process_tree = inductive_miner.apply(refined_log)
            net, initial_marking, final_marking = pt_converter.apply(process_tree)

            print(f"Model discovered: {len(net.places)} places, {len(net.transitions)} transitions")

            print("Mapping transition labels to imprecise labels...")
            for transition in net.transitions:
                if transition.label is not None and transition.label in label_mapping:
                    transition.label = label_mapping[transition.label]

            return compute_metrics_on_model(imprecise_log, net, initial_marking, final_marking)

        except Exception as e:
            print(f"Error in label-mapped evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        return compute_process_mining_metrics(refined_log)


def compute_metrics_on_model(event_log, net, initial_marking, final_marking):
    metrics = {}

    print("Computing fitness...")
    try:
        fitness_result = replay_fitness_evaluator.apply(
            event_log, net, initial_marking, final_marking,
            variant=replay_fitness_evaluator.Variants.TOKEN_BASED
        )
        metrics['fitness'] = fitness_result['average_trace_fitness']
        print(f"  Fitness: {metrics['fitness']:.4f}")
    except Exception as e:
        print(f"  Error computing fitness: {e}")
        metrics['fitness'] = 0.0

    print("Computing precision...")
    try:
        precision = precision_evaluator.apply(
            event_log, net, initial_marking, final_marking,
            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
        )
        metrics['precision'] = precision
        print(f"  Precision: {metrics['precision']:.4f}")
    except Exception as e:
        print(f"  Error computing precision: {e}")
        metrics['precision'] = 0.0

    print("Computing generalization...")
    try:
        generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
        metrics['generalization'] = generalization
        print(f"  Generalization: {metrics['generalization']:.4f}")
    except Exception as e:
        print(f"  Error computing generalization: {e}")
        metrics['generalization'] = 0.0

    print("Computing simplicity...")
    try:
        simplicity = simplicity_evaluator.apply(net)
        metrics['simplicity'] = simplicity
        print(f"  Simplicity: {metrics['simplicity']:.4f}")
    except Exception as e:
        print(f"  Error computing simplicity: {e}")
        metrics['simplicity'] = 0.0

    if metrics['precision'] > 0 and metrics['fitness'] > 0:
        metrics['fscore'] = 2 * (metrics['precision'] * metrics['fitness']) / (metrics['precision'] + metrics['fitness'])
        print(f"  F-Score: {metrics['fscore']:.4f}")
    else:
        metrics['fscore'] = 0.0

    return metrics


def has_ground_truth_labels(gt_df):
    if gt_df.empty:
        return False
    if control_flow_column_ground_truth in gt_df.columns:
        unique_labels = gt_df[control_flow_column_ground_truth].dropna().unique()
        return len(unique_labels) > 1 and 'unknown' not in unique_labels
    return False


def create_ground_truth_log(gt_df):
    try:
        if case_id_column not in gt_df.columns:
            print(f"Warning: Case ID column '{case_id_column}' not found in ground truth")
            return None

        event_log = EventLog()

        for case_id, case_group in gt_df.groupby(case_id_column):
            trace = Trace()
            trace.attributes['case:concept:name'] = str(case_id)

            case_group_sorted = case_group.sort_values(event_id_column)

            for _, row in case_group_sorted.iterrows():
                event = Event()
                activity_name = row.get(control_flow_column_ground_truth, 'unknown')
                event['concept:name'] = activity_name
                event['EventID'] = row[event_id_column]

                if 'SYSCALL_timestamp' in row and pd.notnull(row['SYSCALL_timestamp']):
                    try:
                        event['time:timestamp'] = pd.to_datetime(row['SYSCALL_timestamp'], unit='s')
                    except:
                        pass

                trace.append(event)

            if len(trace) > 0:
                event_log.append(trace)

        print(f"Created ground truth log with {len(event_log)} traces")
        return event_log

    except Exception as e:
        print(f"Error creating ground truth log: {e}")
        return None


def compute_ground_truth_model_metrics(ground_truth_log, imprecise_log):
    try:
        process_tree = inductive_miner.apply(ground_truth_log)
        net, initial_marking, final_marking = pt_converter.apply(process_tree)

        label_mapping = {}
        for trace in imprecise_log:
            for event in trace:
                if 'ground_truth' in event:
                    gt_label = event['ground_truth']
                    imprecise_label = event['concept:name']
                    if gt_label not in label_mapping:
                        label_mapping[gt_label] = imprecise_label

        if not label_mapping:
            for trace in ground_truth_log:
                for event in trace:
                    gt_label = event['concept:name']
                    imprecise_label = gt_label.split('_')[0] if '_' in gt_label else gt_label
                    if gt_label not in label_mapping:
                        label_mapping[gt_label] = imprecise_label

        print(f"Ground truth to imprecise mapping: {len(label_mapping)} labels")

        for transition in net.transitions:
            if transition.label and transition.label in label_mapping:
                transition.label = label_mapping[transition.label]

        return compute_metrics_on_model(imprecise_log, net, initial_marking, final_marking)

    except Exception as e:
        print(f"Error computing ground truth model metrics: {e}")
        return None


def compute_process_mining_metrics(event_log):
    if event_log is None or len(event_log) == 0:
        print("Error: Empty or invalid event log")
        return None

    try:
        print(f"\n=== Process Mining Metrics Calculation ===")
        print(f"Log has {len(event_log)} traces with {sum(len(trace) for trace in event_log)} events total")

        if len(event_log) < 2:
            print("Warning: Too few traces for meaningful process mining analysis")
            return {'fitness': 0.0, 'precision': 0.0, 'generalization': 0.0, 'simplicity': 0.0, 'fscore': 0.0}

        print(f"Discovering process model using Inductive Miner (without noise filtering for consistency with baselines)...")
        process_tree = inductive_miner.apply(event_log)
        net, initial_marking, final_marking = pt_converter.apply(process_tree)

        print(f"Model discovered: {len(net.places)} places, {len(net.transitions)} transitions")

        metrics = {}

        print("Computing fitness...")
        try:
            fitness_result = replay_fitness_evaluator.apply(
                event_log, net, initial_marking, final_marking,
                variant=replay_fitness_evaluator.Variants.TOKEN_BASED
            )
            metrics['fitness'] = fitness_result['average_trace_fitness']
            print(f"  Fitness: {metrics['fitness']:.4f}")
        except Exception as e:
            print(f"  Error computing fitness: {e}")
            metrics['fitness'] = 0.0

        print("Computing precision...")
        try:
            precision = precision_evaluator.apply(
                event_log, net, initial_marking, final_marking,
                variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
            )
            metrics['precision'] = precision
            print(f"  Precision: {metrics['precision']:.4f}")
        except Exception as e:
            print(f"  Error computing precision: {e}")
            metrics['precision'] = 0.0

        print("Computing generalization...")
        try:
            generalization = generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
            metrics['generalization'] = generalization
            print(f"  Generalization: {metrics['generalization']:.4f}")
        except Exception as e:
            print(f"  Error computing generalization: {e}")
            metrics['generalization'] = 0.0

        print("Computing simplicity...")
        try:
            simplicity = simplicity_evaluator.apply(net)
            metrics['simplicity'] = simplicity
            print(f"  Simplicity: {metrics['simplicity']:.4f}")
        except Exception as e:
            print(f"  Error computing simplicity: {e}")
            metrics['simplicity'] = 0.0

        if metrics['precision'] > 0 and metrics['fitness'] > 0:
            metrics['fscore'] = 2 * (metrics['precision'] * metrics['fitness']) / (metrics['precision'] + metrics['fitness'])
            print(f"  F-Score: {metrics['fscore']:.4f}")
        else:
            metrics['fscore'] = 0.0
            print(f"  F-Score: {metrics['fscore']:.4f} (precision or fitness is 0)")

        return metrics

    except Exception as e:
        print(f"Error computing process mining metrics: {e}")
        import traceback
        traceback.print_exc()
        return {'fitness': 0.0, 'precision': 0.0, 'generalization': 0.0, 'simplicity': 0.0, 'fscore': 0.0}


if __name__ == "__main__":
    main()
