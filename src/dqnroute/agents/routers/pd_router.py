import random

from typing import List, Tuple

from numpy import add
from ..base import *
from ...messages import *

class PDRouter(Router):
    """
    Simplest router which just sends a packet in a random direction.
    """
    def __init__(self, adj_links, edge_weight='weight',  **kwargs):
        super().__init__(**kwargs)
        self.network = nx.DiGraph()
        self.edge_weight = edge_weight
        
        self.network.add_node(self.id)
        for (m, params) in adj_links.items():
            self.network.add_edge(self.id, m, **params)

    def addLink(self, to: AgentId, params={}) -> List[Message]:
        msgs = super().addLink(to, params)
        self.network.add_edge(to, self.id, **params)
        self.network.add_edge(self.id, to, **params)
        return msgs + self._announceState()

    def removeLink(self, to: AgentId) -> List[Message]:
        msgs = super().removeLink(to)
        self.network.remove_edge(to, self.id)
        self.network.remove_edge(self.id, to)
        return msgs + self._announceState()

    # def only_reachable(self.network, pkg.dst, self.network.successors(self.id)):
        
    def handle(self, event: WorldEvent) -> List[WorldEvent]:
        if isinstance(event, VehicleStartRouterEvent):
            allowed_nbrs = list(self.network.adj[self.id].keys()) 
            to_nbr, additional_msgs = self.route(allowed_nbrs) 
            return [VehicleRouterAction(to_nbr,event.vehicle)] + additional_msgs
        
        elif isinstance(event, VehicleEnqueuedEvent):
            allowed_nbrs = list(self.network.adj[self.id].keys()) 
            to_nbr, additional_msgs = self.route(allowed_nbrs) 
            return [VehicleRouterAction(to_nbr,event.vehicle)] + additional_msgs
        else:
            return super().handleEvent(event)

    def route(self, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        # print("route", allowed_nbrs, random.choice(allowed_nbrs))
        to = random.choice(allowed_nbrs)
        return to , []
