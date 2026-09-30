"""Agent composition — wire agents into workflows."""
from nexus.compose.pipeline import Pipeline
from nexus.compose.router import Router
from nexus.compose.parallel import Parallel
from nexus.compose.graph import Graph

__all__ = ["Pipeline", "Router", "Parallel", "Graph"]
