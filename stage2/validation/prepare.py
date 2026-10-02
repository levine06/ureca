"""Fetch and freeze only publicly released pre-cutoff development structures."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CUTOFF = "2021-09-30"
records = []
for pdb_id in ("1UBQ", "1CRN", "1LYZ"):
    with urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}") as response:
        metadata = json.load(response)
    released = metadata["rcsb_accession_info"]["initial_release_date"][:10]
    if released >= CUTOFF:
        raise ValueError(f"Refusing post-cutoff entry: {pdb_id}, {released}")
    with urlopen(f"https://files.rcsb.org/download/{pdb_id}.pdb") as response:
        content = response.read()
    (ROOT / f"{pdb_id}.pdb").write_bytes(content)
    records.append({"pdb_id": pdb_id, "release_date": released,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "role": "development_only", "chain": "A"})
(ROOT / "manifest.json").write_text(json.dumps({"cutoff_exclusive": CUTOFF,
    "policy": "Exclude these fixtures from final evaluation; Stage 1 split is pending.",
    "structures": records}, indent=2))
print(json.dumps(records, indent=2))
