"""Shared data contracts. Every service imports from here; nobody redefines these shapes.

See schemas.py for the three contracts:
    1. Event schema       simulator / video adapter -> stats engine
    2. Story atom schema  stats engine              -> agent team
    3. Insight schema     stats engine + agents     -> web app, Teams, overlay API
"""

from koora.contracts.schemas import *  # noqa: F401,F403
from koora.contracts.schemas import SCHEMA_VERSION  # noqa: F401
