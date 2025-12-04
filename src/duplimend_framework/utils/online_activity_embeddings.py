import numpy as np

class DynamicActivityEmbeddings:
    def __init__(self, embedding_dim, seed):
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        self.embeddings = {}

    def get(self, activity):
        """Get embedding for activity, creating if not exists"""
        if activity not in self.embeddings:
            self.embeddings[activity] = self.rng.normal(0, 1, self.embedding_dim).tolist()
        return self.embeddings[activity]
    
    def update(self, activity_label):
        """Update/add activity embedding (creates if not exists)"""
        return self.get(activity_label)
    
    def get_embedding(self, activity_label):
        """Alternative method name for compatibility"""
        return self.get(activity_label)
    
    def add_activity(self, activity_label):
        """Alternative method name for compatibility"""
        return self.get(activity_label)