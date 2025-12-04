import numpy as np
from config.config import feature_selection_params
from src.duplimend_framework.utils.control_flow_feature_utils import get_control_flow_feature_names

class FeatureVectorBuilder:
    @staticmethod
    def _is_null_value(value):
        """Check if a value represents null/missing data"""
        if value is None or value == -1:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "nan", "null", "none", "na"}
        return False
    
    @staticmethod
    def build(event_dict, data_columns, control_flow_features, online_onehot_encoders, activity_embeddings, excluded_columns, use_control_flow_features):
        features = []
        missing_count = 0
        
        for col in data_columns:
            if col in excluded_columns:
                continue
                
            value = event_dict.get(col, None)
            
            if col in online_onehot_encoders:
                if col not in excluded_columns: 
                    features.extend(online_onehot_encoders[col].encode(value))
            else:
                if FeatureVectorBuilder._is_null_value(value):
                    features.append(0.0)
                    missing_count += 1
                else:
                    try:
                        features.append(float(value))
                    except (TypeError, ValueError):
                        features.append(0.0)
                        missing_count += 1

        if use_control_flow_features and control_flow_features is not None:
            cf_boost = feature_selection_params.get("control_flow_feature_boost", 1.0)
            for cf_name in get_control_flow_feature_names():
                cf_value = control_flow_features.get(cf_name, None)
                if cf_name.startswith("prev_"):
                    if (activity_embeddings and
                        hasattr(activity_embeddings, 'embeddings') and
                        len(activity_embeddings.embeddings) > 0 and
                        cf_value is not None):
                        emb = activity_embeddings.get(cf_value)
                        if emb is None:
                            emb = [0.0] * activity_embeddings.embedding_dim
                    else:
                        embedding_dim = getattr(activity_embeddings, 'embedding_dim', 8) if activity_embeddings else 8
                        emb = [0.0] * embedding_dim

                    features.extend([v * cf_boost for v in emb])
                else:
                    if FeatureVectorBuilder._is_null_value(cf_value):
                        features.append(0.0)
                    else:
                        try:
                            features.append(float(cf_value) * cf_boost)
                        except (TypeError, ValueError):
                            features.append(0.0)
                            
        return np.array(features, dtype=np.float32)