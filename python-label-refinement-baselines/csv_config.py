#!/usr/bin/env python3
"""
CSV Configuration for Baseline Approaches
Enables CSV input with configurable column mappings
"""

# CSV Column Configuration
CSV_CONFIG = {
    # Basic column mappings - EDIT THESE TO MATCH YOUR CSV COLUMNS
    "case_id_column": "case:concept:name",      # Case ID column name
    "activity_column": "concept:name",          # Activity column name  
    "timestamp_column": "time:timestamp",       # Timestamp column name
    "resource_column": "org:resource",          # Resource column name (optional)
    
    # Alternative configurations for different datasets
    "cybersecurity_config": {
        "case_id_column": "SYSCALL_pid",
        "activity_column": "activity_label", 
        "timestamp_column": "SYSCALL_timestamp",
        "resource_column": "PROCESS_comm"
    },

    "ipalia_config": {
        "case_id_column": "case:concept:name",
        "activity_column": "concept:name",
        "timestamp_column": "time:timestamp",
        "resource_column": None
    },

    "document_review_config": {
        "case_id_column": "CaseID",
        "activity_column": "Activity",
        "timestamp_column": "Timestamp",
        "resource_column": "Resource"
    },

    # Default configuration selector
    "active_config": "document_review_config"  # Options: "default", "cybersecurity_config", "mining_config", "healthcare_config", "ipalia_config", "document_review_config"
}

def get_column_mapping(config_name="default"):
    """Get column mapping for specified configuration"""
    if config_name == "default":
        return {
            "case_id": CSV_CONFIG["case_id_column"],
            "activity": CSV_CONFIG["activity_column"],
            "timestamp": CSV_CONFIG["timestamp_column"],
            "resource": CSV_CONFIG.get("resource_column")
        }
    else:
        config = CSV_CONFIG.get(config_name, {})
        return {
            "case_id": config.get("case_id_column", "case:concept:name"),
            "activity": config.get("activity_column", "concept:name"),
            "timestamp": config.get("timestamp_column", "time:timestamp"),
            "resource": config.get("resource_column")
        }

def get_active_column_mapping():
    """Get the currently active column mapping"""
    active_config = CSV_CONFIG.get("active_config", "default")
    return get_column_mapping(active_config)

def print_column_mappings():
    """Print all available column mappings"""
    print("Available CSV Column Configurations:")
    print("="*50)
    
    for config_name, config in CSV_CONFIG.items():
        if isinstance(config, dict) and config_name != "active_config":
            print(f"\n{config_name}:")
            if config_name == "default":
                print(f"  Case ID: {CSV_CONFIG['case_id_column']}")
                print(f"  Activity: {CSV_CONFIG['activity_column']}")
                print(f"  Timestamp: {CSV_CONFIG['timestamp_column']}")
                print(f"  Resource: {CSV_CONFIG.get('resource_column', 'None')}")
            else:
                print(f"  Case ID: {config.get('case_id_column', 'N/A')}")
                print(f"  Activity: {config.get('activity_column', 'N/A')}")
                print(f"  Timestamp: {config.get('timestamp_column', 'N/A')}")
                print(f"  Resource: {config.get('resource_column', 'None')}")
    
    print(f"\nActive Configuration: {CSV_CONFIG.get('active_config', 'default')}")

if __name__ == "__main__":
    print_column_mappings()