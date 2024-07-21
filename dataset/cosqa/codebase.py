import json

codebase = {}
for file in ['cosqa-retrieval-dev-500.json', 'cosqa-retrieval-test-500.json']: 
    for js in json.load(open(file)): 
        codebase[js['code']] = js['retrieval_idx']

with open('codebase_idx_map.txt', 'w') as f: 
    json.dump(codebase, f, indent=0)
