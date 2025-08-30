#!/usr/bin/env python3
"""
Materials Recovery Analyzer
Analyzes all materials files to identify salvageable data and standardization opportunities
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

class MaterialsRecoveryAnalyzer:
    """Analyze and recover materials data from various schema formats"""
    
    def __init__(self, materials_dir: str):
        self.materials_dir = Path(materials_dir)
        self.schema_patterns = {}
        self.salvageable_materials = {}
        self.critical_properties = [
            'melting', 'boiling', 'curie', 'transition', 'decomposition',
            'sublimation', 'glass', 'critical', 'superconducting'
        ]
    
    def load_file_safely(self, filepath: Path) -> Optional[Dict]:
        """Safely load JSON file with multiple encoding attempts"""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                    # Clean up common issues
                    content = content.replace('\x00', '')  # Remove null bytes
                    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)  # Remove control chars
                    return json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except Exception as e:
                print(f"Error with {filepath}: {e}")
                continue
        
        return None
    
    def identify_schema_pattern(self, data: Dict, filename: str) -> str:
        """Identify the schema pattern used in the data"""
        if not isinstance(data, dict):
            if isinstance(data, list):
                return "array_format"
            return "unknown"
        
        # Check for standard schema (our target format)
        if 'material' in data and 'properties' in data:
            props = data['properties']
            if isinstance(props, dict) and any(k in props for k in ['physical', 'chemical', 'thermal']):
                return "standard_schema"
        
        # Check for flat property format (like old Fe.json)
        if any(key.endswith('Properties') for key in data.keys()):
            return "flat_properties"
        
        # Check for pageTitle format (like H2.json was)
        if 'pageTitle' in data or 'data' in data:
            return "page_title_format"
        
        # Check for section-based array format
        if isinstance(data, dict) and 'section' in str(data):
            return "section_array"
        
        # Check for simple key-value pairs
        if all(isinstance(v, (str, int, float)) for v in data.values()):
            return "simple_kv"
        
        return "complex_nested"
    
    def extract_material_name(self, data: Dict, filename: str) -> str:
        """Extract material name from various formats"""
        # Standard format
        if 'material' in data:
            return data['material']
        
        # Page title format
        if 'pageTitle' in data:
            return data['pageTitle']
        
        # Try to infer from filename
        name = filename.replace('.json', '').replace('_', ' ')
        
        # Clean up common patterns
        name = re.sub(r'^[A-Z][a-z]*_', '', name)  # Remove prefix like "Al_"
        name = re.sub(r'_\d+', '', name)  # Remove numeric suffixes
        
        return name
    
    def extract_temperatures(self, data: Dict, schema_pattern: str) -> Dict[str, float]:
        """Extract temperature values from various schema formats"""
        temperatures = {}
        
        def extract_temp_from_string(text: str) -> Optional[float]:
            """Extract temperature value from string"""
            if not text:
                return None
            
            # Look for temperature patterns
            patterns = [
                r'(-?\d+\.?\d*)\s*°?[CF]',
                r'(-?\d+\.?\d*)\s*C',
                r'(-?\d+\.?\d*)\s*F',
                r'(-?\d+\.?\d*)\s*K',
                r'(-?\d+\.?\d*)\s*deg',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        temp = float(match.group(1))
                        # Convert to Celsius if needed
                        if 'F' in text.upper():
                            temp = (temp - 32) * 5/9
                        elif 'K' in text.upper():
                            temp = temp - 273.15
                        return temp
                    except ValueError:
                        continue
            return None
        
        def search_for_temp_properties(obj, path=""):
            """Recursively search for temperature properties"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = key.lower()
                    
                    # Check if key indicates temperature property
                    if any(temp_word in key_lower for temp_word in self.critical_properties):
                        if isinstance(value, (str, dict)):
                            temp_val = None
                            
                            if isinstance(value, str):
                                temp_val = extract_temp_from_string(value)
                            elif isinstance(value, dict):
                                # Check nested values
                                for sub_key, sub_val in value.items():
                                    if isinstance(sub_val, str):
                                        temp_val = extract_temp_from_string(sub_val)
                                        if temp_val is not None:
                                            break
                            
                            if temp_val is not None:
                                temperatures[key] = temp_val
                    
                    # Recurse into nested structures
                    search_for_temp_properties(value, f"{path}.{key}" if path else key)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_temp_properties(item, f"{path}[{i}]")
        
        search_for_temp_properties(data)
        return temperatures
    
    def analyze_all_materials(self) -> Dict:
        """Analyze all materials files"""
        print("=== Materials Recovery Analysis ===")
        print("Scanning all materials files...\n")
        
        results = {
            'schema_patterns': {},
            'salvageable_count': 0,
            'total_files': 0,
            'materials_with_temps': 0,
            'recovered_materials': {}
        }
        
        # Get all JSON files
        json_files = list(self.materials_dir.glob("*.json"))
        results['total_files'] = len(json_files)
        
        for filepath in json_files:
            filename = filepath.name
            print(f"Analyzing {filename}...")
            
            # Load file
            data = self.load_file_safely(filepath)
            if data is None:
                print(f"  ✗ Could not load {filename}")
                continue
            
            # Identify schema
            schema = self.identify_schema_pattern(data, filename)
            results['schema_patterns'][schema] = results['schema_patterns'].get(schema, 0) + 1
            
            # Extract material info
            material_name = self.extract_material_name(data, filename)
            temperatures = self.extract_temperatures(data, schema)
            
            # Check if salvageable
            is_salvageable = len(temperatures) > 0 or schema == "standard_schema"
            
            if is_salvageable:
                results['salvageable_count'] += 1
                if temperatures:
                    results['materials_with_temps'] += 1
                
                results['recovered_materials'][filename] = {
                    'material_name': material_name,
                    'schema_pattern': schema,
                    'temperatures_found': temperatures,
                    'temp_count': len(temperatures),
                    'data_preview': str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                }
                
                print(f"  ✓ Salvageable: {material_name}")
                if temperatures:
                    for temp_type, temp_val in temperatures.items():
                        print(f"    • {temp_type}: {temp_val:.1f}°C")
            else:
                print(f"  ✗ No useful data found")
            print()
        
        # Print summary
        print("=== Recovery Summary ===")
        print(f"Total files scanned: {results['total_files']}")
        print(f"Salvageable materials: {results['salvageable_count']}")
        print(f"Materials with temperatures: {results['materials_with_temps']}")
        print("\nSchema patterns found:")
        for schema, count in results['schema_patterns'].items():
            print(f"  {schema}: {count} files")
        
        return results
    
    def generate_standardization_plan(self, analysis_results: Dict) -> List[str]:
        """Generate plan for standardizing salvageable materials"""
        plan = []
        
        recoverable = analysis_results['recovered_materials']
        
        for filename, info in recoverable.items():
            if info['schema_pattern'] != 'standard_schema':
                priority = "HIGH" if info['temp_count'] > 0 else "MEDIUM"
                plan.append(f"{priority}: Standardize {filename} ({info['material_name']}) - {info['temp_count']} temps found")
        
        return sorted(plan, key=lambda x: x.startswith('HIGH'), reverse=True)

if __name__ == "__main__":
    analyzer = MaterialsRecoveryAnalyzer("/mnt/d/Git_repo/E3_Project-main/E3_Engine/materials_data")
    results = analyzer.analyze_all_materials()
    
    # Generate standardization plan
    plan = analyzer.generate_standardization_plan(results)
    
    if plan:
        print("\n=== Standardization Plan ===")
        for item in plan:
            print(f"  {item}")
    
    # Save detailed results
    output_file = "/mnt/d/Git_repo/E3_Project-main/materials_recovery_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed analysis saved to: {output_file}")