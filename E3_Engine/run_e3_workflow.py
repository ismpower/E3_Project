import json, os
from datetime import datetime

os.makedirs('data/processed', exist_ok=True) 
os.makedirs('results/reports', exist_ok=True)

cases = [{'id': f'BYU_{b}G_{t}K', 'temp': t, 'field': b, 'anomaly': b>0} 
         for t in [48,96] for b in [0,50,100,150,200]]

with open('data/processed/dataset.json', 'w') as f:
    json.dump({'cases': cases, 'count': len(cases)}, f)

print(f"✅ E3 Engine ready! Created {len(cases)} cases")
