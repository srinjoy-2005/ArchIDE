#!/usr/bin/env bash
# ==============================================================================
# ArchIDE: Compile Agentic IR (.ir.json) to Visual Graphs (.arch) with Validation
# ==============================================================================
# Usage:
#   ./scripts/compile_all_to_arch.sh                  # Compiles and validates all IR files in workspace/ir/
#   ./scripts/compile_all_to_arch.sh <file.ir.json>   # Compiles and validates a single IR file
#   ./scripts/compile_all_to_arch.sh --no-validate    # Compiles without validation pass
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

echo -e "${BOLD}${CYAN}=== ArchIDE: Compile IR (.ir.json) -> .arch (with Shape & PyTorch Validation) ===${NC}"

if [ "$#" -eq 0 ]; then
    echo -e "${YELLOW}Compiling and validating all IR graphs in workspace/ir/ to workspace/graphs/...${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" --all-to-arch
elif [ "$1" = "--no-validate" ] && [ "$#" -eq 1 ]; then
    echo -e "${YELLOW}Compiling all IR graphs in workspace/ir/ to workspace/graphs/ (validation skipped)...${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" --all-to-arch --no-validate
else
    echo -e "${YELLOW}Compiling specified target(s): $@${NC}\n"
    python "${PROJECT_ROOT}/backend/agent_compiler.py" "$@" --to-arch
fi

echo -e "\n${BOLD}${GREEN}✔ Architecture compilation & validation completed successfully.${NC}"
