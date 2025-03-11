import os
import json
from typing import Tuple, List

def get_dataset() -> Tuple[List[str], List[str]]:
    dataset_path = os.environ["dataset_path"]
    sources, targets = [], []
    with open(dataset_path) as f:
        for line in f:
            line = json.loads(line.strip())
            sources.append(line["text"])
            targets.append(line["summaries"][0])

    return sources, targets
