#!/usr/bin/env bash
# Validate all ```mermaid blocks in project markdown (requires Node + mermaid-cli).
#
# Usage:
#   ./scripts/validate_mermaid.sh
#   MERMAID_CLI=/path/to/mmdc ./scripts/validate_mermaid.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMDC="${MERMAID_CLI:-}"

if [[ -z "${MMDC}" ]]; then
  if command -v mmdc >/dev/null 2>&1; then
    MMDC="$(command -v mmdc)"
  elif [[ -x "${ROOT}/.tools/mermaid-cli/node_modules/.bin/mmdc" ]]; then
    MMDC="${ROOT}/.tools/mermaid-cli/node_modules/.bin/mmdc"
  else
    echo "Installing @mermaid-js/mermaid-cli into ${ROOT}/.tools/mermaid-cli ..."
    mkdir -p "${ROOT}/.tools/mermaid-cli"
    npm install --prefix "${ROOT}/.tools/mermaid-cli" @mermaid-js/mermaid-cli@11.4.0 >/dev/null
    MMDC="${ROOT}/.tools/mermaid-cli/node_modules/.bin/mmdc"
  fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail=0
while IFS= read -r -d '' file; do
  python3 - "${file}" "${TMP}" "${MMDC}" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

file_path = Path(sys.argv[1])
tmp = Path(sys.argv[2])
mmdc = sys.argv[3]
text = file_path.read_text(encoding="utf-8")
for index, match in enumerate(re.finditer(r"```mermaid\n(.*?)\n```", text, re.S), start=1):
    body = match.group(1) + "\n"
    stem = f"{file_path.name.replace('.', '_')}_{index}"
    inp = tmp / f"{stem}.mmd"
    out = tmp / f"{stem}.svg"
    inp.write_text(body, encoding="utf-8")
    result = subprocess.run([mmdc, "-i", str(inp), "-o", str(out)], capture_output=True, text=True)
    rel = file_path
    try:
        rel = file_path.relative_to(Path.cwd())
    except ValueError:
        pass
    if result.returncode != 0:
        print(f"FAIL {rel} block #{index}")
        print(result.stderr.strip())
        sys.exit(1)
    print(f"OK   {rel} block #{index}")
PY
  if [[ $? -ne 0 ]]; then
    fail=1
  fi
done < <(find "${ROOT}" -name '*.md' \
  -not -path '*/.data/*' \
  -not -path '*/.venv/*' \
  -not -path '*/node_modules/*' \
  -print0)

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "All mermaid blocks validated."
