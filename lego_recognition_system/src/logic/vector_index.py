import numpy as np
import pickle
import os
from scipy.spatial.distance import cdist

class VectorIndex:
    def __init__(self, index_path=None):
        self.embeddings = []
        self.metadata = [] # List of {ldraw_id: ...}
        self.index_path = index_path
        
        if index_path and os.path.exists(index_path):
            self.load(index_path)

    def add(self, embedding, metadata):
        self.embeddings.append(embedding)
        self.metadata.append(metadata)

    def search(self, query_embedding, k=1):
        if not self.embeddings:
            return []
            
        # Reshape query if single
        query = query_embedding.reshape(1, -1)
        db = np.stack(self.embeddings)
        
        # Calculate Cosine Distance (1 - cosine similarity)
        # Cosine similarity is better for embeddings
        distances = cdist(query, db, metric='cosine')[0]
        
        # Get indices of top k smallest distances
        top_k_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'metadata': self.metadata[idx],
                'distance': float(distances[idx]),
                'similarity': 1.0 - float(distances[idx])
            })
            
        return results

    def save(self, path):
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Vector index saved with {len(self.embeddings)} entries to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            # Use extending to allow cumulative loading
            self.embeddings.extend(data['embeddings'])
            self.metadata.extend(data['metadata'])
        print(f"📂 Loaded {len(data['embeddings'])} vectors from {os.path.basename(path)}. Total database: {len(self.embeddings)}")
