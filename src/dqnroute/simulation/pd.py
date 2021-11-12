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

logger = logging.getLogger(PD_LOGGER)

class PDRouterFactory(HandlerFactory):
    def __init__(self, router_type, routers_cfg, context = None,
                 topology_graph = None, training_router_type = None, **kwargs):

        RouterClass = get_router_class(router_type, context)
        self.context = context
        self.router_cfg = routers_cfg.get(router_type, {})
        self.edge_weight = 'latency' if context == 'network' else 'length'
        self._dyn_env = None


        self.training_mode = False
        self.router_type = router_type
        self.RouterClass = RouterClass
        super().__init__(**kwargs)

        if topology_graph is None:
            self.topology_graph = self.conn_graph.to_directed()
        else:
            self.topology_graph = topology_graph


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

        if issubclass(self.RouterClass, PDRouter):
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

    def __init__(self, data_series: EventSeries, requests: List, vehicles_param, **kwargs):
        self.data_series = data_series
        super().__init__(**kwargs)
        self.requests= requests

        self.vehicles= []
        for i, vp in enumerate(vehicles_param):
            self.vehicles.append(Vehicle(id = i,
                                        capacity= vp['capacity'],
                                        max_road= vp['max_road'],
                                        road_till_now= 0,
                                        current_amount= 0,
                                        current_pos= ('router', vp['v']),
                                        env= self.env))
        

    def makeConnGraph(self, network_cfg, **kwargs) -> nx.Graph:
        return make_pd_graph(network_cfg)

    def makeHandlerFactory(self, **kwargs):
        return PDRouterFactory(router_type='pd_router', context='pd', **kwargs)


    def handleAction(self, from_agent: AgentId, action: Action) -> Event:
        """
        discuss case of VehicleRouteAction, VehicleReceiveAction
        """

        if isinstance(action, VehicleRouterAction):
            to_agent = action.to
            if not self.conn_graph.has_edge(from_agent, to_agent):
                raise Exception("Trying to route to a non-neighbor")
            
            edge_params = self.conn_graph[from_agent][to_agent]
            distance =  edge_params['weight']
            vehicle= action.vehicle
            if vehicle.road_till_now + distance >= vehicle.max_road:
                raise Exception(f"Vehicle_{vehicle.id}'s gas out.")
            
            self.env.process(self._vehicleTransfer(from_agent, to_agent, action.vehicle))
            return Event(self.env).succeed()
        
    def _vehicleTransfer(self, from_agent: AgentId, to_agent: AgentId, vehicle: Vehicle):
        """
        Transfer Vehicle from one agent to another.
        """
        logger.debug("Vehicle #{} transports: {} -> {} with amount: {} and free capacity: {}, road till now: {}, remaining road: {}".format(
            vehicle.id, 
            from_agent[1], 
            to_agent[1],
            vehicle.current_amount,
            vehicle.capacity - vehicle.current_amount,
            vehicle.road_till_now,
            vehicle.max_road - vehicle.road_till_now))
        edge_params = self.conn_graph[from_agent][to_agent]
        distance =  edge_params['weight']
        vehicle.road_till_now += distance
        with vehicle.request() as req:
            yield req
            yield self.env.timeout(distance)
            print(self.env.now)
            
        self.handleWorldEvent(VehicleEnqueuedEvent(to_agent, vehicle))

    def handleWorldEvent(self, event: WorldEvent) -> Event:
        if isinstance(event, VehicleEnqueuedEvent):
            self.env.process(self._processInVehicle(event.agent, event.vehicle))
            return self.passToAgent(event.agent, event)
        else:
            return super().handleWorldEvent(event)

    def _processInVehicle(self, agent: AgentId, vehicle: Vehicle):
        """
        Process the loading process inside node.
        """
        for request in self.requests:
            if request['v'] == agent[1]:
                demand = request['demands']
                yield self.env.timeout(abs(demand))
                # case it's a demand 
                if demand < 0:
                    if vehicle.current_amount > abs(demand):
                        vehicle.current_amount += demand
                    else:
                        raise Exception('Not Enough Amount in Viehvle') 
                
                else:
                    if vehicle.current_amount + demand > vehicle.capacity:
                        raise Exception('Not Enough Space in Viehvle') 
                    else:
                        vehicle.current_amount += demand
                self.requests.remove(request)
                break
                # case it's a pick up
        
        

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
                            requests= self.run_params['requests'],
                            network_cfg=self.run_params['network'],
                            routers_cfg=self.run_params['settings']['router'],
                            vehicles_param = self.run_params['vehicles'],
                            **kwargs)

    def relevantConfig(self):
        return self.run_params

    def makeRunId(self, random_seed):
        return '{}-{}'.format(self.world.factory.router_type, random_seed)

    def runProcess(self, random_seed = None):

        if random_seed is not None:
            set_random_seed(random_seed)

        all_nodes = list(self.world.conn_graph.nodes)
        all_edges = list(self.world.conn_graph.edges(data=True))

        for vehicle in self.world.vehicles:
            yield self.world.handleWorldEvent(VehicleEnqueuedEvent(all_nodes[vehicle.current_pos[1]], vehicle))

                
