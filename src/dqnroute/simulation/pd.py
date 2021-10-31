import logging
import pprint
import math
import numpy as np
import networkx as nx

from typing import List, Callable, Dict, Tuple
from functools import reduce
from simpy import Environment, Event, Resource, Process, Interrupt
from ..utils import *
from ..conveyor_model import *
from ..messages import *
from ..event_series import *
from ..agents import *
from ..constants import *
from .common import *


class PDRouterFactory(HandlerFactory):
    def __init__(self, router_type, routers_cfg, context = None,
                 topology_graph = None, training_router_type = None, **kwargs):
        RouterClass = get_router_class(router_type, context)
        self.context = context
        self.router_cfg = routers_cfg.get(router_type, {})
        self.edge_weight = 'latency' if context == 'network' else 'length'
        self._dyn_env = None

        if training_router_type is None:
            self.training_mode = False
            self.router_type = router_type
            self.RouterClass = RouterClass
        else:
            self.training_mode = True
            TrainerClass = get_router_class(training_router_type, context)
            self.router_type = 'training__{}__{}'.format(router_type, training_router_type)
            self.RouterClass = TrainingRouterClass(RouterClass, TrainerClass, **kwargs)
        super().__init__(**kwargs)

        if topology_graph is None:
            self.topology_graph = self.conn_graph.to_directed()
        else:
            self.topology_graph = topology_graph

        if self.training_mode:
            dummy = RouterClass(
                **self._handlerArgs(('router', 0), neighbours=[], random_init=True))
            self.brain = dummy.brain
            self.router_cfg['brain'] = self.brain

    def dynEnv(self):
        if self._dyn_env is None:
            return DynamicEnv(time=lambda: self.env.now)
        else:
            return self._dyn_env

    def useDynEnv(self, env):
        self._dyn_env = env

    def makeMasterHanlder(self) -> MasterHandler:
        dyn_env = self.dynEnv()
        return self.RouterClass(
            env=dyn_env, network=self.topology_graph,
            edge_weight='latency', **self.router_cfg)

    def _handlerArgs(self, agent_id, **kwargs):
        G = self.topology_graph
        kwargs.update({
            'env': self.dynEnv(),
            'id': agent_id,
            'edge_weight': self.edge_weight,
            'nodes': sorted(list(G.nodes())),
            'edges_num': len(G.edges()), # small hack to make link-state initialization simpler
        })
        kwargs.update(self.router_cfg)

        if issubclass(self.RouterClass, LinkStateRouter):
            kwargs['adj_links'] = G.adj[agent_id]
        return kwargs

    def makeHandler(self, agent_id: AgentId, **kwargs) -> MessageHandler:
        assert agent_id[0] == 'router', "Only routers are allowed in computer network"
        return self.RouterClass(**self._handlerArgs(agent_id, **kwargs))

    def centralized(self):
        return issubclass(self.RouterClass, MasterHandler)

class PDEnviroment(MultiAgentEnv):
    """
    Environment which models the store enviroment and vehicles and commidity trasformation.
    """
    context = 'pd'

    def __init__(self, data_series: EventSeries, **kwargs):
        self.data_series = data_series
        super().__init__(**kwargs)

    def makeConnGraph(self, network_cfg, **kwargs) -> nx.Graph:
        return make_pd_graph(network_cfg)

    def makeHandlerFactory(self, **kwargs):
        return PDRouterFactory(context='network', **kwargs)

    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        pass

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        pass

    def _edgeTransfer(self, from_agent: AgentId, to_agent: AgentId, pkg: Package):
        pass

    def _inputQueue(self, from_agent: AgentId, to_agent: AgentId, pkg: Package):
        pass

class PDRunner(SimulationRunner):
    """
    Class which constructs and runs scenarios in PickUp and Delivery simulation
    environment.
    """
    context = 'network'

    def __init__(self, data_dir=LOG_DATA_DIR+'/PD', **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)

    def makeDataSeries(self, series_period, series_funcs):
        return event_series(series_period, series_funcs)

    def makeMultiAgentEnv(self, **kwargs) -> MultiAgentEnv:
        """
        build our enviroment.
        """
        return PDEnviroment(env=self.env, 
                            data_series=self.data_series,
                            network_cfg=self.run_params['network'],
                            routers_cfg=self.run_params['settings']['router'],
                            **kwargs)

    def relevantConfig(self):
        ps = self.run_params
        return ps

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.world.factory.router_type, random_seed)

    def runProcess(self, random_seed = None):
        raise NotImplementedError