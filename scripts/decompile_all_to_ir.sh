#!/usr/bin/env bash
# ==============================================================================
# ArchIDE: Decompile Visual .arch Graphs to Agentic IR (.ir.json)
# ==============================================================================
# Usage:
#   ./scripts/decompile_all_to_ir.sh                  # Decompiles all .arch files to workspace/ir/
#   ./scripts/decompile_all_to_ir.sh <file.arch>      # Decompiles a single .arch file
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

echo -e "${BOLD}${CYAN}=== ArchIDE: Decompile .arch -> IR (.ir.json) ===${NC}"

if [ "$#" -eq 0 ]; then
    echo -e "${YELLOW}Decompiling all graphs in workspace/graphs/ to workspace/ir/...${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" --all-to-ir
else
    echo -e "${YELLOW}Decompiling specified target(s): $@${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" "$@" --to-ir
fi

echo -e "\n${BOLD}${GREEN}✔ IR decompile completed successfully.${NC}"
