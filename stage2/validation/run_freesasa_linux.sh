#!/usr/bin/env bash
set -euo pipefail
validation_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage2_dir="$(dirname -- "$validation_dir")"
if [[ ! -f "$stage2_dir/pyproject.toml" ]]; then
    echo "Place this script in stage2/validation, beside the Stage 2 package." >&2
    exit 1
fi
cd -- "$stage2_dir"
python3 -m venv .venv-freesasa-linux
test_python="$stage2_dir/.venv-freesasa-linux/bin/python"
"$test_python" -m pip install --upgrade pip
"$test_python" -m pip install -e '.[test]'
"$test_python" -m pip install 'freesasa==2.2.1' --use-pep517 --no-cache-dir
"$test_python" -c 'import freesasa; print(freesasa.__file__)'
"$test_python" -m pip freeze > "$validation_dir/freesasa-linux-environment.txt"
"$test_python" -m pytest tests/test_physical_sanity.py -k freesasa -v -rs \
    --junitxml="$validation_dir/freesasa-linux-results.xml" \
    | tee "$validation_dir/freesasa-linux-log.txt"
"$test_python" - "$validation_dir/freesasa-linux-results.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases) == 4, f'Expected four tests, found {len(cases)}'
assert all(case.find('skipped') is None and case.find('failure') is None
           and case.find('error') is None for case in cases)
print('All four FreeSASA tests passed without skips.')
PY
