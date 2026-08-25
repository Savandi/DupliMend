import matplotlib.pyplot as plt
from collections import defaultdict
import time
import threading


class ClusterMonitor:
    def __init__(self, cluster_managers):
        self.cluster_managers = cluster_managers
        self.history = defaultdict(lambda: defaultdict(list))
        self.split_count = defaultdict(int)
        self.merge_count = defaultdict(int)
        self.timestamp_history = []
        self.monitoring = False

    def start_monitoring(self, interval=5):
        """Start periodic monitoring in a background thread"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        self.monitoring = False

    def _monitor_loop(self, interval):
        while self.monitoring:
            self.capture_snapshot()
            time.sleep(interval)

    def capture_snapshot(self):
        """Record the current state of all clusters"""
        current_time = time.time()
        self.timestamp_history.append(current_time)

        for activity, manager in self.cluster_managers.items():
            cluster_count = 0
            for adapter in manager.clustering_adapters.values():
                significant_clusters = adapter.get_significant_clusters(manager.min_cluster_weight)
                cluster_count += len(significant_clusters)

            self.history[activity]["cluster_count"].append(cluster_count)

        print(f"[MONITOR] Cluster counts: {dict((a, self.history[a]['cluster_count'][-1]) for a in self.history)}")

    def record_split(self, activity):
        self.split_count[activity] += 1

    def record_merge(self, activity):
        self.merge_count[activity] += 1

    def plot_evolution(self):
        """Generate a plot showing how clusters have evolved over time"""
        plt.figure(figsize=(12, 8))

        for activity, data in self.history.items():
            if "cluster_count" in data and len(data["cluster_count"]) > 1:
                plt.plot(
                    range(len(data["cluster_count"])),
                    data["cluster_count"],
                    label=f"{activity} (clusters: {data['cluster_count'][-1]})"
                )

        plt.title("Cluster Evolution Over Time")
        plt.xlabel("Monitoring Intervals")
        plt.ylabel("Cluster Count")
        plt.legend()
        plt.grid(True)

        plt.savefig("cluster_evolution.png")
        print(f"[MONITOR] Evolution plot saved to cluster_evolution.png")

        print("\n=== CLUSTER OPERATION SUMMARY ===")
        for activity in sorted(set(list(self.split_count.keys()) + list(self.merge_count.keys()))):
            print(f"{activity}: {self.split_count[activity]} splits, {self.merge_count[activity]} merges")
