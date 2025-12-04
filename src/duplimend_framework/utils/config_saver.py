"""
Config saving utility for DupliMend framework.
Provides consistent config saving across all offline execution stages.
"""

import os
import shutil
import json
from datetime import datetime
import sys

def save_config_to_output(output_dir, stage_name="general", additional_info=None):
    """
    Save current config.py to output directory for reproducibility.
    
    Args:
        output_dir (str): Directory where config should be saved
        stage_name (str): Name of the execution stage (e.g., "feature_generation", "autoencoder_training", "main_execution")
        additional_info (dict): Optional additional metadata to save
        
    Returns:
        str: Path to saved config file, or None if failed
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Find config.py file - try multiple possible locations
        possible_config_paths = [
            "config/config.py",
            "config.py", 
            os.path.join(os.path.dirname(__file__), "../../../config/config.py"),
            os.path.join(os.getcwd(), "config/config.py")
        ]
        
        config_source = None
        for path in possible_config_paths:
            if os.path.exists(path):
                config_source = os.path.abspath(path)
                break
        
        if not config_source:
            return None
        
        # Save config file with stage-specific name
        config_dest = os.path.join(output_dir, f"config_{stage_name}_{timestamp}.py")
        shutil.copy2(config_source, config_dest)
        
        # Try to extract config values for metadata
        config_values = {}
        try:
            # Add project root to path to import config
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            import config.config as config_module
            
            # Extract relevant config values using getattr to avoid import * issues
            config_values = {
                'feature_vectors_base_dir': getattr(config_module, 'feature_vectors_base_dir', 'N/A'),
                'models_save_dir': getattr(config_module, 'models_save_dir', 'N/A'),
                'streaming_config': getattr(config_module, 'streaming_config', {}),
                'evaluation_config': getattr(config_module, 'evaluation_config', {}),
                'training_mode_config': getattr(config_module, 'training_mode_config', {}),
                'autoencoder_params': getattr(config_module, 'autoencoder_params', {}),
                'clustering_params': getattr(config_module, 'clustering_params', {}),
            }
            
        except Exception:
            pass
        
        # Create comprehensive metadata
        metadata = {
            'execution_stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'config_source': config_source,
            'config_saved_to': config_dest,
            'working_directory': os.getcwd(),
            'output_directory': output_dir,
            'python_executable': sys.executable,
            'config_values': config_values,
            'additional_info': additional_info or {}
        }
        
        # Save metadata file
        metadata_file = os.path.join(output_dir, f"execution_info_{stage_name}_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return config_dest

    except Exception:
        return None

def load_config_metadata(output_dir, stage_name=None):
    """
    Load previously saved config metadata from output directory.
    
    Args:
        output_dir (str): Directory where config was saved
        stage_name (str): Optional stage name to filter results
        
    Returns:
        list: List of metadata dictionaries
    """
    try:
        if not os.path.exists(output_dir):
            return []
        
        metadata_files = []
        for file in os.listdir(output_dir):
            if file.startswith("execution_info_") and file.endswith(".json"):
                if stage_name is None or stage_name in file:
                    metadata_files.append(os.path.join(output_dir, file))
        
        metadata_list = []
        for file_path in sorted(metadata_files):
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    metadata_list.append(metadata)
            except Exception:
                continue

        return metadata_list

    except Exception:
        return []

def verify_config_consistency(output_dir, stages=None):
    """
    Verify that configs used across different stages are consistent.
    
    Args:
        output_dir (str): Directory where configs were saved
        stages (list): Optional list of stages to check
        
    Returns:
        bool: True if configs are consistent, False otherwise
    """
    try:
        if stages is None:
            stages = ["feature_generation", "autoencoder_training", "main_execution"]
        
        configs = {}
        for stage in stages:
            metadata_list = load_config_metadata(output_dir, stage)
            if metadata_list:
                # Get the most recent config for this stage
                latest_metadata = max(metadata_list, key=lambda x: x.get('timestamp', ''))
                configs[stage] = latest_metadata.get('config_values', {})
        
        if len(configs) < 2:
            return True
        
        # Compare configs
        reference_stage = list(configs.keys())[0]
        reference_config = configs[reference_stage]
        
        inconsistent = False
        for stage, config in configs.items():
            if stage == reference_stage:
                continue
                
            for key, value in reference_config.items():
                if key in config and config[key] != value:
                    inconsistent = True

        return not inconsistent

    except Exception:
        return False

if __name__ == "__main__":
    test_dir = "/tmp/test_config_saver"
    save_config_to_output(test_dir, "test_stage")
    metadata = load_config_metadata(test_dir)