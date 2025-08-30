#!/usr/bin/env python3
"""
Comprehensive Bruno Framework Validation
Tests Bruno constant against curated, validated materials
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from bruno_framework.theory.bruno_threshold import bruno_threshold_check, KAPPA_VALIDATED

class ComprehensiveBrunoValidator:
    """Validate Bruno framework against literature-accurate materials"""
    
    def __init__(self, materials_dir: str):
        self.materials_dir = Path(materials_dir)
        self.validated_materials = [
            "Carbon.json",
            "Tungsten.json", 
            "Fe.json",
            "Diamond.json",
            "Diamond_Synthetic.json",
            "Diamond_Polycrystalline.json",
            "H2.json",
            "Chlorine.json",
            "Ni.json",
            "Pyrolytic_Graphite.json",
            "3Al2O3-2GeO2.json",
            "Al2O3_orundum_99.9.json",
            "Br2.json",
            "TaC.json"
        ]
    
    def load_material_data(self, filename: str) -> dict:
        """Load validated material data"""
        filepath = self.materials_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return {}
    
    def extract_critical_temperatures(self, material_data: dict) -> dict:
        """Extract temperature-related properties"""
        temps = {}
        properties = material_data.get('properties', {})
        
        # Check thermal properties
        thermal_props = properties.get('thermal', [])
        for prop in thermal_props:
            prop_name = prop.get('property', '').lower()
            if 'melting' in prop_name or 'boiling' in prop_name or 'curie' in prop_name:
                metric_val = prop.get('metric', '')
                # Extract numeric temperature (simple parsing)
                try:
                    temp_str = metric_val.split()[0].replace(',', '')
                    if temp_str.replace('-', '').replace('.', '').isdigit():
                        temps[prop.get('property')] = float(temp_str)
                except:
                    pass
        
        return temps
    
    def calculate_bruno_parameters(self, temperatures: dict, material_name: str) -> dict:
        """Calculate Bruno parameters for given temperatures"""
        results = {}
        
        for temp_type, temp_celsius in temperatures.items():
            temp_kelvin = temp_celsius + 273.15 if temp_celsius > -273 else abs(temp_celsius)
            
            # Calculate Bruno parameter β_B = κ × T
            beta_b = KAPPA_VALIDATED * temp_kelvin
            
            # Determine transition prediction
            is_critical = beta_b >= 1.0
            
            results[temp_type] = {
                'temperature_c': temp_celsius,
                'temperature_k': temp_kelvin,
                'beta_b': beta_b,
                'is_critical_transition': is_critical,
                'material': material_name
            }
        
        return results
    
    def validate_against_literature(self, bruno_results: dict) -> dict:
        """Validate Bruno predictions against known physics"""
        validation = {}
        
        for temp_type, result in bruno_results.items():
            material = result['material']
            temp_c = result['temperature_c']
            beta_b = result['beta_b']
            is_critical = result['is_critical_transition']
            
            # Known critical temperatures for validation
            expected_critical = False
            if 'melting' in temp_type.lower():
                expected_critical = True  # Melting is always a phase transition
            elif 'curie' in temp_type.lower():
                expected_critical = True  # Magnetic transition
            elif 'boiling' in temp_type.lower():
                expected_critical = True  # Liquid-gas transition
            
            # Check if Bruno prediction matches expected
            prediction_correct = is_critical == expected_critical
            
            validation[temp_type] = {
                'material': material,
                'temperature': temp_c,
                'bruno_parameter': beta_b,
                'bruno_predicts_critical': is_critical,
                'literature_is_critical': expected_critical,
                'prediction_correct': prediction_correct
            }
        
        return validation
    
    def run_comprehensive_validation(self) -> dict:
        """Run full validation suite"""
        print("=== Comprehensive Bruno Framework Validation ===")
        print(f"Using Bruno constant κ = {KAPPA_VALIDATED} K⁻¹")
        print()
        
        all_results = {}
        validation_summary = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'materials_tested': 0,
            'critical_temps_found': 0
        }
        
        for material_file in self.validated_materials:
            print(f"Analyzing {material_file}...")
            
            # Load material data
            material_data = self.load_material_data(material_file)
            if not material_data:
                continue
            
            material_name = material_data.get('material', material_file.replace('.json', ''))
            validation_summary['materials_tested'] += 1
            
            # Extract temperatures
            temperatures = self.extract_critical_temperatures(material_data)
            if not temperatures:
                print(f"  No critical temperatures found for {material_name}")
                continue
            
            validation_summary['critical_temps_found'] += len(temperatures)
            
            # Calculate Bruno parameters
            bruno_results = self.calculate_bruno_parameters(temperatures, material_name)
            
            # Validate predictions
            validation = self.validate_against_literature(bruno_results)
            
            # Update summary
            for temp_type, result in validation.items():
                validation_summary['total_predictions'] += 1
                if result['prediction_correct']:
                    validation_summary['correct_predictions'] += 1
            
            all_results[material_name] = validation
            
            # Print results for this material
            print(f"  Material: {material_name}")
            for temp_type, result in validation.items():
                status = "✓" if result['prediction_correct'] else "✗"
                print(f"    {status} {temp_type}: {result['temperature']:.1f}°C, β_B={result['bruno_parameter']:.3f}")
            print()
        
        # Print summary
        print("=== Validation Summary ===")
        print(f"Materials tested: {validation_summary['materials_tested']}")
        print(f"Critical temperatures found: {validation_summary['critical_temps_found']}")
        print(f"Total predictions: {validation_summary['total_predictions']}")
        if validation_summary['total_predictions'] > 0:
            accuracy = validation_summary['correct_predictions'] / validation_summary['total_predictions'] * 100
            print(f"Correct predictions: {validation_summary['correct_predictions']} ({accuracy:.1f}%)")
        
        return {
            'results': all_results,
            'summary': validation_summary,
            'bruno_constant': KAPPA_VALIDATED
        }

if __name__ == "__main__":
    validator = ComprehensiveBrunoValidator("/mnt/d/Git_repo/E3_Project-main/E3_Engine/materials_data")
    results = validator.run_comprehensive_validation()
    
    # Save results
    output_file = "/mnt/d/Git_repo/E3_Project-main/validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")