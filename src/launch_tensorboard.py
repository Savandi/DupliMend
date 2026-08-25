import os
import sys
import subprocess
import time
import shutil

def clean_tensorboard_environment():
    """Clean TensorBoard environment completely"""
    import tempfile
    
    paths_to_clean = [
        os.path.join(tempfile.gettempdir(), '.tensorboard-info'),
        os.path.expanduser('~/.tensorboard'),
        os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'tensorboard'),
    ]
    
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print(f"[CLEAN] Removed: {path}")
            except Exception as e:
                print(f"[CLEAN] Could not remove {path}: {e}")

def kill_existing_tensorboard():
    """Kill any existing TensorBoard processes"""
    try:
        subprocess.run(['taskkill', '/f', '/im', 'tensorboard.exe'], 
                      capture_output=True, check=False)
        subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                      capture_output=True, check=False)
        time.sleep(2)
    except:
        pass

def launch_tensorboard_isolated(logdir, port=6007):
    """Launch TensorBoard in isolated environment"""
    
    print("[CLEAN] Cleaning TensorBoard environment...")
    kill_existing_tensorboard()
    clean_tensorboard_environment()
    
    time.sleep(3)
    
    if not os.path.exists(logdir):
        print(f"[ERROR] Log directory doesn't exist: {logdir}")
        return False
    
    contents = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                contents.append(os.path.join(root, file))
    
    if not contents:
        print(f"[WARNING] No TensorBoard event files found in: {logdir}")
    else:
        print(f"[INFO] Found {len(contents)} TensorBoard event files")
    
    cmd = [
        sys.executable, "-m", "tensorboard",
        "--logdir", logdir,
        "--port", str(port),
        "--reload_interval", "5",
        "--bind_all",
        "--samples_per_plugin", "scalars=0",
    ]
    
    print(f"[LAUNCH] Starting TensorBoard...")
    print(f"[LAUNCH] Command: {' '.join(cmd)}")
    print(f"[LAUNCH] Logdir: {logdir}")
    print(f"[LAUNCH] Port: {port}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        
        time.sleep(5)
        
        if process.poll() is None:
            print(f"[SUCCESS] TensorBoard started successfully!")
            print(f"[ACCESS] http://localhost:{port}")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"[ERROR] TensorBoard failed to start:")
            print(f"[STDOUT] {stdout.decode()}")
            print(f"[STDERR] {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to launch TensorBoard: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python launch_tensorboard_clean.py <logdir> [port]")
        print("Example: python launch_tensorboard_clean.py \"C:\\path\\to\\logs\" 6007")
        return
    
    logdir = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 6007
    
    print(f"[CONFIG] Logdir: {logdir}")
    print(f"[CONFIG] Port: {port}")
    
    success = launch_tensorboard_isolated(logdir, port)
    
    if success:
        print("\n" + "="*60)
        print("TENSORBOARD LAUNCHED SUCCESSFULLY")
        print("="*60)
        print(f"Access TensorBoard at: http://localhost:{port}")
        print("Press Ctrl+C to stop TensorBoard")
        print("="*60)
        
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] TensorBoard stopped by user")
    else:
        print("\n[FAILED] Could not launch TensorBoard")

if __name__ == "__main__":
    main()