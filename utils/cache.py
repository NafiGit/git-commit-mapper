import os
import hashlib
import pickle
from datetime import datetime

class AnalysisCache:
    """Cache for storing analysis results."""
    
    def __init__(self, cache_dir=".analysis_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load existing cache from disk."""
        cache_file = os.path.join(self.cache_dir, "analysis_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached file analyses")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        cache_file = os.path.join(self.cache_dir, "analysis_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_file_hash(self, file_path):
        """Calculate hash of file contents."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return None
    
    def get_analysis(self, file_path, commit_id):
        """Get cached analysis if available."""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return None
        
        cache_key = f"{commit_id}:{file_path}"
        if cache_key in self.cache and self.cache[cache_key]['hash'] == file_hash:
            return self.cache[cache_key]['result']
        return None
    
    def store_analysis(self, file_path, commit_id, result):
        """Store analysis result in cache."""
        file_hash = self.get_file_hash(file_path)
        if file_hash:
            cache_key = f"{commit_id}:{file_path}"
            self.cache[cache_key] = {
                'hash': file_hash,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }