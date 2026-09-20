#!/usr/bin/env bash
# ==============================================================================
# ArchIDE: Compile Visual Graphs (.arch) to PyTorch Code (.py)
# ==============================================================================
# Usage:
#   ./scripts/compile_all_to_python.sh                 # Compiles all .arch files in workspace/graphs/ to workspace/python/
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

echo -e "${BOLD}${CYAN}=== ArchIDE: Compile .arch -> PyTorch (.py) ===${NC}"

echo -e "${YELLOW}Compiling all visual graphs in workspace/graphs/ to workspace/python/...${NC}\n"
python "${PROJECT_ROOT}/backend/agent_compiler.py" --all-to-python

echo -e "\n${BOLD}${GREEN}✔ PyTorch compilation completed successfully.${NC}"
