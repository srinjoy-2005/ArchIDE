import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from blocks import get_all_blocks

def get_block_registry_schema():
    schema = {}
    for block_instance in get_all_blocks():
        definition = block_instance.definition
        block_id = definition.id
        
        block_schema = {
            "name": definition.name,
            "category": definition.category,
            "is_functional": definition.is_functional,
            "inputs": [{"id": p.id, "name": p.name, "is_list": getattr(p, 'is_list', False)} for p in definition.inputs],
            "outputs": [{"id": p.id, "name": p.name} for p in definition.outputs],
            "params": []
        }
        
        for p in definition.params:
            param_dict = {
                "name": p.name,
                "type": p.type,
                "default": p.default,
                "description": p.description
            }
            block_schema["params"].append(param_dict)
            
        schema[block_id] = block_schema
        
    return schema

if __name__ == "__main__":
    registry = get_block_registry_schema()
    out_path = os.path.join(os.path.dirname(__file__), "block_schema.json")
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Dumped {len(registry)} blocks to {out_path}")
