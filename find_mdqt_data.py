#!/usr/bin/env python3
import os
from pathlib import Path

def find_mdqt_outputs(search_dir):
    """Find all MDQT simulation outputs"""
    search_path = Path(search_dir)
    
    print(f"Searching for MDQT data in: {search_path.absolute()}")
    
    # Look for .dat files
    dat_files = list(search_path.rglob("*.dat"))
    if dat_files:
        print(f"\nFound {len(dat_files)} .dat files:")
        for f in dat_files[:10]:  # Show first 10
            print(f"  {f.relative_to(search_path)}")
        if len(dat_files) > 10:
            print(f"  ... and {len(dat_files) - 10} more")
    
    # Look for job directories
    job_dirs = []
    for root, dirs, files in os.walk(search_path):
        for d in dirs:
            if 'job' in d.lower():
                job_dirs.append(Path(root) / d)
    
    if job_dirs:
        print(f"\nFound {len(job_dirs)} job directories:")
        for d in job_dirs:
            print(f"  {d.relative_to(search_path)}")
            # Check contents
            contents = list(d.glob("*.dat"))
            if contents:
                print(f"    Contains {len(contents)} .dat files")
    
    # Look for simulation parameter directories
    sim_dirs = []
    for root, dirs, files in os.walk(search_path):
        for d in dirs:
            if any(x in d.lower() for x in ['ge', 'density', 'ions']):
                sim_dirs.append(Path(root) / d)
    
    if sim_dirs:
        print(f"\nFound {len(sim_dirs)} simulation directories:")
        for d in sim_dirs:
            print(f"  {d.relative_to(search_path)}")
    
    # Try to find the most recent energies.dat
    energies_files = list(search_path.rglob("energies.dat"))
    if energies_files:
        print(f"\nFound energies.dat files:")
        for f in energies_files:
            print(f"  {f.relative_to(search_path)} (size: {f.stat().st_size} bytes)")
            return f.parent  # Return the directory containing energies.dat
    
    return None

if __name__ == "__main__":
    result_dir = find_mdqt_outputs("external/mdqt-sim")
    
    if result_dir:
        print(f"\n✅ Found MDQT output in: {result_dir}")
        
        # Test our processor
        print("\n🧪 Testing MDQT processor...")
        try:
            import sys
            sys.path.append('e3_engine')
            from mdqt_processor import MDQTProcessor
            
            processor = MDQTProcessor(result_dir)
            processor.load_simulation_data()
            print("✅ Data loaded successfully")
            
            features = processor.extract_physics_features()
            print(f"✅ Features extracted: {len(features)} parameters")
            
            # Show a few key features
            for key in ['density', 'temperature', 'anomaly_score']:
                if key in features:
                    print(f"  {key}: {features[key]:.3e}")
            
        except Exception as e:
            print(f"❌ Processor test failed: {e}")
    else:
        print("\n❌ No MDQT output found")
