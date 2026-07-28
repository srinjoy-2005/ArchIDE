from models import BlockDef, PortDef, ParamDef
from blocks import get_all_block_defs

# Expose REGISTRY as a list of BlockDef objects for the frontend
REGISTRY = get_all_block_defs()

