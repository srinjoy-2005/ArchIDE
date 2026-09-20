#!/usr/bin/env bash
# ==============================================================================
# ArchIDE: Decompile PyTorch Code (.py) to Agentic IR (.ir.json)
# ==============================================================================
# Usage:
#   ./scripts/decompile_python_to_ir.sh                 # Decompiles all .py files in workspace/python/ to workspace/ir/
#   ./scripts/decompile_python_to_ir.sh <file.py>       # Decompiles a single .py file
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Activate virtualenv if available
if [ -f "${PROJECT_ROOT}/backend/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/backend/.venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Terminal colors
BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}=== ArchIDE: Decompile PyTorch (.py) -> IR (.ir.json) ===${NC}"

if [ "$#" -eq 0 ]; then
    echo -e "${YELLOW}Decompiling all Python models in workspace/python/ to workspace/ir/...${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" --all-from-python
else
    echo -e "${YELLOW}Decompiling specified target(s): $@${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" "$@" --from-python
fi

echo -e "\n${BOLD}${GREEN}✔ PyTorch decompile completed successfully.${NC}"
