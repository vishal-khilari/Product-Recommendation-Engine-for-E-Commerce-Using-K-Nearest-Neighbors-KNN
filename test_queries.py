import requests
import json

hostname = '127.0.0.1'

# Search test
queries = ["Home First Aid Kit", "Trauma / Bleeding Control Kit", "Burn Care Kit"]
for q in queries:
    res = requests.get(f"http://{hostname}:5000/api/recommend?q={q}")
    if res.status_code == 200:
        data = res.json()
        print(f"Query: {q}")
        print(f"  Target: {data.get('target', {}).get('name')[:50]}")
    else:
        print(f"Query: {q} failed: {res.text}")
