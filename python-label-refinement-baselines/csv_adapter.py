#!/usr/bin/env python3
"""
CSV to XES Adapter for Baseline Approaches
Converts CSV files to XES format with configurable column mappings
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from csv_config import get_active_column_mapping, get_column_mapping, print_column_mappings

class CSVToXESAdapter:
    def __init__(self, config_name=None):
        """Initialize adapter with column configuration"""
        if config_name:
            self.column_mapping = get_column_mapping(config_name)
        else:
            self.column_mapping = get_active_column_mapping()
        
        print(f"[CSV_ADAPTER] Using column mapping:")
        print(f"  Case ID: {self.column_mapping['case_id']}")
        print(f"  Activity: {self.column_mapping['activity']}")
        print(f"  Timestamp: {self.column_mapping['timestamp']}")
        print(f"  Resource: {self.column_mapping['resource']}")
    
    def csv_to_xes(self, csv_path, output_xes_path=None, case_id_column=None, activity_column=None, timestamp_column=None):
        """
        Convert CSV file to XES format
        
        Args:
            csv_path: Path to input CSV file
            output_xes_path: Path for output XES file (optional)
            case_id_column: Override case ID column (optional)
            activity_column: Override activity column (optional) 
            timestamp_column: Override timestamp column (optional)
        
        Returns:
            Path to generated XES file
        """
        
        print(f"[CSV_ADAPTER] Converting {csv_path} to XES format...")
        
        # Override column mappings if provided
        columns = {
            'case_id': case_id_column or self.column_mapping['case_id'],
            'activity': activity_column or self.column_mapping['activity'],
            'timestamp': timestamp_column or self.column_mapping['timestamp'],
            'resource': self.column_mapping['resource']
        }
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
            print(f"[CSV_ADAPTER] Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"[CSV_ADAPTER] Available columns: {list(df.columns)}")
        except Exception as e:
            raise Exception(f"Failed to read CSV file: {e}")
        
        # Validate required columns exist
        required_cols = ['case_id', 'activity', 'timestamp']
        missing_cols = []
        
        for col_type in required_cols:
            col_name = columns[col_type]
            if col_name not in df.columns:
                missing_cols.append(f"{col_type} ({col_name})")
        
        if missing_cols:
            print(f"[ERROR] Missing columns: {missing_cols}")
            print(f"[ERROR] Available columns: {list(df.columns)}")
            raise Exception(f"Missing required columns: {missing_cols}")
        
        # Convert to PM4Py EventLog
        event_log = EventLog()
        
        # Group by case ID
        case_groups = df.groupby(columns['case_id'])
        
        for case_id, case_df in case_groups:
            trace = Trace()
            trace.attributes['case:concept:name'] = str(case_id)
            
            # Sort by timestamp if possible
            try:
                case_df_sorted = case_df.sort_values(columns['timestamp'])
            except:
                print(f"[WARNING] Could not sort by timestamp for case {case_id}")
                case_df_sorted = case_df
            
            # Create events
            for _, row in case_df_sorted.iterrows():
                event = Event()
                
                # Standard XES attributes
                event['concept:name'] = str(row[columns['activity']])
                
                # Add timestamp if possible
                try:
                    timestamp_val = row[columns['timestamp']]
                    if pd.notnull(timestamp_val):
                        if isinstance(timestamp_val, str):
                            # Try to parse string timestamp
                            try:
                                event['time:timestamp'] = pd.to_datetime(timestamp_val)
                            except:
                                print(f"[WARNING] Could not parse timestamp: {timestamp_val}")
                        else:
                            event['time:timestamp'] = pd.to_datetime(timestamp_val)
                except Exception as e:
                    print(f"[WARNING] Timestamp processing error: {e}")
                
                # Add resource if available
                if columns['resource'] and columns['resource'] in df.columns:
                    resource_val = row[columns['resource']]
                    if pd.notnull(resource_val):
                        event['org:resource'] = str(resource_val)
                
                # Add all other columns as event attributes
                for col in df.columns:
                    if col not in [columns['case_id'], columns['activity'], columns['timestamp'], columns['resource']]:
                        val = row[col]
                        if pd.notnull(val):
                            event[f'custom:{col}'] = str(val)
                
                trace.append(event)
            
            if len(trace) > 0:
                event_log.append(trace)
        
        print(f"[CSV_ADAPTER] Created EventLog with {len(event_log)} traces")
        total_events = sum(len(trace) for trace in event_log)
        print(f"[CSV_ADAPTER] Total events: {total_events}")
        
        # Generate output path if not provided
        if not output_xes_path:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            output_dir = os.path.dirname(csv_path)
            output_xes_path = os.path.join(output_dir, f"{base_name}_converted.xes")
        
        # Export to XES
        try:
            xes_exporter.apply(event_log, output_xes_path)
            print(f"[CSV_ADAPTER] Exported XES to: {output_xes_path}")
            return output_xes_path
        except Exception as e:
            raise Exception(f"Failed to export XES file: {e}")
    
    def convert_folder(self, input_folder, output_folder=None):
        """
        Convert all CSV files in a folder to XES format
        
        Args:
            input_folder: Folder containing CSV files
            output_folder: Output folder for XES files (optional)
        
        Returns:
            List of converted XES file paths
        """
        
        if not output_folder:
            output_folder = os.path.join(input_folder, "xes_converted")
        
        os.makedirs(output_folder, exist_ok=True)
        
        converted_files = []
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.csv'):
                csv_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]
                xes_path = os.path.join(output_folder, f"{base_name}.xes")
                
                try:
                    self.csv_to_xes(csv_path, xes_path)
                    converted_files.append(xes_path)
                except Exception as e:
                    print(f"[ERROR] Failed to convert {filename}: {e}")
        
        return converted_files

def main():
    """Command line interface for CSV to XES conversion"""
    if len(sys.argv) < 2:
        print("CSV to XES Adapter")
        print("==================")
        print("Usage:")
        print("  python csv_adapter.py <csv_file> [output_xes_file] [config_name]")
        print("  python csv_adapter.py --folder <input_folder> [output_folder] [config_name]")
        print("  python csv_adapter.py --configs")
        print("")
        print("Examples:")
        print("  python csv_adapter.py data.csv")
        print("  python csv_adapter.py data.csv output.xes")
        print("  python csv_adapter.py data.csv output.xes cybersecurity_config")
        print("  python csv_adapter.py --folder ./csv_data")
        print("  python csv_adapter.py --configs")
        return 1
    
    if sys.argv[1] == "--configs":
        print_column_mappings()
        return 0
    
    if sys.argv[1] == "--folder":
        # Folder conversion mode
        if len(sys.argv) < 3:
            print("Error: Please provide input folder path")
            return 1
        
        input_folder = sys.argv[2]
        output_folder = sys.argv[3] if len(sys.argv) > 3 else None
        config_name = sys.argv[4] if len(sys.argv) > 4 else None
        
        adapter = CSVToXESAdapter(config_name)
        converted_files = adapter.convert_folder(input_folder, output_folder)
        
        print(f"\nConverted {len(converted_files)} files:")
        for file_path in converted_files:
            print(f"  {file_path}")
        
    else:
        # Single file conversion mode
        csv_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        config_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist")
            return 1
        
        adapter = CSVToXESAdapter(config_name)
        
        try:
            xes_path = adapter.csv_to_xes(csv_file, output_file)
            print(f"\nSuccess! Converted to: {xes_path}")
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())