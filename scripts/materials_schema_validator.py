#!/usr/bin/env python3
"""
Materials Data Schema Validator
Validates and standardizes material property data for Bruno framework
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class MaterialsSchemaValidator:
    """Validates and standardizes materials data schema"""
    
    def __init__(self, materials_dir: str):
        self.materials_dir = Path(materials_dir)
        self.standard_schema = {
            "material": "str",
            "properties": {
                "physical": "array",
                "chemical": "array", 
                "mechanical": "array",
                "electrical": "array",
                "thermal": "array",
                "optical": "array",
                "component_elements": "array",
                "descriptive": "object"
            }
        }
    
    def load_material_file(self, filepath: Path) -> Optional[Dict]:
        """Load material data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def validate_file_structure(self, filepath: Path) -> bool:
        """Validate file has required structure"""
        data = self.load_material_file(filepath)
        if not data:
            return False
            
        # Check for material name
        if "material" not in data:
            print(f"Missing 'material' field in {filepath}")
            return False
            
        # Check for properties structure
        if "properties" not in data:
            print(f"Missing 'properties' field in {filepath}")
            return False
            
        return True
    
    def scan_materials_directory(self) -> Dict[str, Any]:
        """Scan materials directory for validation issues"""
        issues = {
            "missing_extensions": [],
            "invalid_json": [],
            "schema_violations": [],
            "empty_files": [],
            "valid_files": []
        }
        
        for filepath in self.materials_dir.glob("*"):
            if filepath.is_file():
                # Check extension
                if not filepath.suffix == '.json':
                    issues["missing_extensions"].append(str(filepath))
                    continue
                
                # Check if empty
                if filepath.stat().st_size < 10:
                    issues["empty_files"].append(str(filepath))
                    continue
                
                # Validate structure
                if self.validate_file_structure(filepath):
                    issues["valid_files"].append(str(filepath))
                else:
                    issues["schema_violations"].append(str(filepath))
        
        return issues
    
    def generate_report(self) -> str:
        """Generate validation report"""
        issues = self.scan_materials_directory()
        
        report = ["Materials Data Validation Report", "=" * 40, ""]
        
        for category, files in issues.items():
            if files:
                report.append(f"{category.replace('_', ' ').title()}:")
                for file in files:
                    report.append(f"  - {file}")
                report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    validator = MaterialsSchemaValidator("/mnt/d/Git_repo/E3_Project-main/E3_Engine/materials_data")
    print(validator.generate_report())