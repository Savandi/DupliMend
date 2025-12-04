import os
import glob
import csv

def diagnose_csv_file_deep(file_path):
    """Deep diagnosis of CSV file - checks all rows for issues"""
    print(f"\nDiagnosing: {file_path}")
    
    try:
        # Check file size
        size = os.path.getsize(file_path)
        print(f"File size: {size} bytes")
        
        if size == 0:
            print("ERROR: File is empty")
            return False
        
        # Check for null bytes
        with open(file_path, 'rb') as f:
            content = f.read()
            null_count = content.count(b'\x00')
            print(f"Null bytes found: {null_count}")
            
            if null_count > 0:
                print("WARNING: File contains null bytes (might be binary)")
        
        # Try different encodings with deep row checking
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin-1']
        successful_encoding = None
        
        for enc in encodings:
            print(f"\nTesting encoding: {enc}")
            try:
                problematic_rows = []
                total_rows = 0
                successful_rows = 0
                
                with open(file_path, 'r', encoding=enc, errors='replace', newline='') as f:
                    csv_reader = csv.DictReader(f)
                    
                    # Check header
                    try:
                        headers = csv_reader.fieldnames
                        print(f"Headers ({len(headers)}): {headers[:5]}{'...' if len(headers) > 5 else ''}")
                    except Exception as e:
                        print(f"ERROR reading headers: {e}")
                        continue
                    
                    # Check each row
                    for row_num, row_dict in enumerate(csv_reader, start=2):  # Start at 2 (header is row 1)
                        total_rows += 1
                        
                        try:
                            # Test if row can be processed
                            if not any(row_dict.values()):
                                problematic_rows.append((row_num, "Empty row"))
                                continue
                            
                            # Check for problematic characters
                            for col, value in row_dict.items():
                                if value and isinstance(value, str):
                                    # Check for null bytes in string values
                                    if '\x00' in value:
                                        problematic_rows.append((row_num, f"Null byte in column '{col}'"))
                                        break
                                    
                                    # Check for extremely long values
                                    if len(value) > 10000:
                                        problematic_rows.append((row_num, f"Very long value in column '{col}' ({len(value)} chars)"))
                                        break
                            
                            successful_rows += 1
                            
                            # Progress update for large files
                            if total_rows % 10000 == 0:
                                print(f"  Checked {total_rows} rows...")
                                
                        except Exception as e:
                            problematic_rows.append((row_num, f"Row processing error: {e}"))
                            continue
                
                print(f"Encoding {enc}: Total rows={total_rows}, Successful={successful_rows}, Problematic={len(problematic_rows)}")
                
                if problematic_rows:
                    print(f"Problematic rows (showing first 10):")
                    for row_num, issue in problematic_rows[:10]:
                        print(f"  Row {row_num}: {issue}")
                    if len(problematic_rows) > 10:
                        print(f"  ... and {len(problematic_rows) - 10} more problematic rows")
                
                if successful_rows > 0 and successful_encoding is None:
                    successful_encoding = enc
                    
            except Exception as e:
                print(f"Encoding {enc}: FAILED - {e}")
                continue
        
        if successful_encoding:
            print(f"RECOMMENDED ENCODING: {successful_encoding}")
            return True
        else:
            print("ERROR: No encoding worked")
            return False
            
    except Exception as e:
        print(f"ERROR: Could not diagnose file - {e}")
        return False

def diagnose_csv_file_quick(file_path):
    """Quick diagnosis - just first few lines"""
    print(f"\nQuick diagnosis: {os.path.basename(file_path)}")
    
    try:
        size = os.path.getsize(file_path)
        print(f"File size: {size} bytes")
        
        if size == 0:
            print("ERROR: File is empty")
            return False
        
        # Try different encodings
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'latin-1']
        
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc, errors='replace') as f:
                    lines = f.readlines()[:5]
                    print(f"Encoding {enc}: SUCCESS - {len(lines)} lines read")
                    if lines:
                        print(f"First line: {repr(lines[0][:100])}")
                        return True
            except Exception as e:
                print(f"Encoding {enc}: FAILED - {e}")
        
        return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def diagnose_folder(folder_path, file_pattern="*.csv", deep_check=False):
    """Diagnose all CSV files in a folder"""
    print(f"Diagnosing all CSV files in folder: {folder_path}")
    print(f"Deep check: {deep_check}")
    print("=" * 60)
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        return
    
    # Get all CSV files
    pattern = os.path.join(folder_path, file_pattern)
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    csv_files.sort()
    print(f"Found {len(csv_files)} CSV files")
    
    successful_files = []
    problematic_files = []
    all_problematic_details = []  # Store all problematic details
    
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {os.path.basename(file_path)}")
        
        if deep_check:
            result = diagnose_csv_file_deep(file_path)
        else:
            result = diagnose_csv_file_quick(file_path)
        
        if result:
            successful_files.append(file_path)
        else:
            problematic_files.append(file_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(csv_files)}")
    print(f"Successful files: {len(successful_files)}")
    print(f"Problematic files: {len(problematic_files)}")
    
    if problematic_files:
        print(f"\nProblematic files:")
        for file_path in problematic_files:
            print(f"  - {os.path.basename(file_path)}")
        
        print(f"\n*** RUN INDIVIDUAL DEEP CHECK ON PROBLEMATIC FILES FOR DETAILS ***")
        print(f"*** Total problematic files: {len(problematic_files)} out of {len(csv_files)} ***")
    else:
        print("\n*** ALL FILES PASSED VALIDATION ***")
    
    return successful_files, problematic_files

# Run quick check first
# print("=== QUICK CHECK ===")
# diagnose_folder("U:/Research/Projects/sef/stream_quality_drift/processed_train_data", deep_check=True)

# Deep check (slower but finds all problematic rows):
if __name__ == "__main__":
    print("=== DEEP CHECK ===")
    successful, problematic = diagnose_folder("U:/Research/Projects/sef/stream_quality_drift/processed_train_data", deep_check=True)
    
    if problematic:
        print("\n" + "="*60)
        print("DETAILED ANALYSIS OF PROBLEMATIC FILES")
        print("="*60)
        
        for file_path in problematic:
            print(f"\n--- DETAILED CHECK: {os.path.basename(file_path)} ---")
            diagnose_csv_file_deep(file_path)