from functools import total_ordering
from copy import deepcopy
from typing import Tuple

from .utils import AgentId, InterfaceId

##
# Elementary datatypes
#

class WorldEvent:
    """
    Utility class, which allows access to arbitrary attrs defined
    at object creation. Base class for `Message` and `Action`.
    """

    def __init__(self, **kwargs):
        self.contents = kwargs

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.contents))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __getattr__(self, name):
        try:
            return super().__getattribute__('contents')[name]
        except KeyError:
            raise AttributeError(name)

    def getContents(self):
        return self.contents


class Message(WorldEvent):
    """
    Event which is a message from some other agent
    """
    pass

class Action(WorldEvent):
    """
    Event which is some agent's action in the physical world
    """
    pass

class UnsupportedEventType(Exception):
    """
    Exception which is thrown by event handlers on encounter of
    unknown event type
    """
    pass

class UnsupportedMessageType(Exception):
    """
    Exception which is thrown by message handlers on encounter of
    unknown message type
    """
    pass

class UnsupportedActionType(Exception):
    """
    Exception which is thrown by the environment on encounter of
    unknown agent action type
    """
    pass

##
# Delays/timeouts
#

class DelayedEvent(WorldEvent):
    """
    Special wrapper event which should be handled not immediately,
    but after some time. Models the timeout logic in agents themselves.
    """
    def __init__(self, id: int, delay: float, inner: WorldEvent):
        super().__init__(id=id, delay=delay, inner=inner)

class DelayInterrupt(WorldEvent):
    """
    Special event which is used to un-schedule the handling of
    `DelayedEvent`
    """
    def __init__(self, delay_id: int):
        super().__init__(delay_id=delay_id)

##
# Core messages, handled by `ConnectionModel`
#

class WireTransferMsg(Message):
    """
    Message which has a payload and a number of interface which it relates to
    (which it came from or which it is going to).
    The `ConnectionModel` deals only with those messages. Interface id -1 is a
    loopback interface.
    """
    def __init__(self, interface: InterfaceId, payload: Message):
        super().__init__(interface=interface, payload=payload)

class WireOutMsg(WireTransferMsg):
    pass

class WireInMsg(WireTransferMsg):
    pass


##
# Basic message classes on a `MessageHandler` level.
#

class InitMessage(Message):
    """
    Message which router receives as environment starts
    """
    def __init__(self, config):
        super().__init__(config=config)

class TransferMessage(Message):
    """
    Wrapper message which is used to send data between nodes
    """
    def __init__(self, from_node: AgentId, to_node: AgentId, inner_msg: Message):
        super().__init__(from_node=from_node, to_node=to_node, inner_msg=inner_msg)

class InMessage(TransferMessage):
    """
    Wrapper message which has came from the outside.
    """
    pass

class OutMessage(TransferMessage):
    """
    Wrapper message which is sent to a neighbor through the interface
    with given ID.
    """
    pass

class DelayTriggerMsg(Message):
    """
    Utility message which is meant to be used only
    with `DelayedEvent`, so that agent can plan some actions
    for some time in the future
    """
    def __init__(self, delay_id: int):
        super().__init__(delay_id=delay_id)

class ServiceMessage(Message):
    """
    Message which does not contain a package and, hence,
    contains no destination.
    """
    pass

class EventMessage(Message):
    """
    Message which contains `Event` which should be handled
    """
    def __init__(self, event: WorldEvent):
        super().__init__(event=event)

class ActionMessage(Message):
    """
    Message which contains `Action` which an agent should perform.
    """
    def __init__(self, action: Action):
        super().__init__(action=action)

class SlaveEvent(WorldEvent):
    """
    An event detected by a slave controller with given ID
    """
    def __init__(self, slave_id: AgentId, inner: WorldEvent):
        super().__init__(slave_id=slave_id, inner=inner)

##
# Computer network events/actions
#

# Packages
@total_ordering
class Package:
    def __init__(self, pkg_id, size, dst, start_time, contents):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.contents = contents
        # self.route = None
        # self.rnn_state = (np.zeros((1, state_size)),
        #                   np.zeros((1, state_size)))

    # def route_add(self, data, cols):
    #     if self.route is None:
    #         self.route = pd.DataFrame(columns=cols)
    #     self.route.loc[len(self.route)] = data

    def __str__(self):
        return '{}#{}{}'.format(self.__class__.__name__, self.id,
                                str((self.dst, self.size, self.start_time, self.contents)))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __hash__(self):
        return hash((self.id, self.contents))

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

class PkgEnqueuedEvent(WorldEvent):
    """
    Some package got enqueued into the router
    """
    def __init__(self, sender: AgentId, recipient: AgentId, pkg: Package):
        super().__init__(sender=sender, recipient=recipient, pkg=pkg)

class PkgProcessingEvent(WorldEvent):
    """
    Some package is now ready to be routed further
    """
    def __init__(self, sender: AgentId, recipient: AgentId, pkg: Package):
        super().__init__(sender=sender, recipient=recipient, pkg=pkg)

class PkgReceiveAction(Action):
    """
    A destination router has received a package.
    """
    def __init__(self, pkg: Package):
        super().__init__(pkg=pkg)

class PkgRouteAction(Action):
    """
    Router has re-routed a package to a neighbour
    """
    def __init__(self, to: AgentId, pkg: Package):
        super().__init__(to=to, pkg=pkg)

class LinkUpdateEvent(WorldEvent):
    """
    A link between nodes has appeared or disappeared
    """
    def __init__(self, u: AgentId, v: AgentId, **kwargs):
        super().__init__(u=u, v=v, **kwargs)

class AddLinkEvent(LinkUpdateEvent):
    """
    Event which router receives when a link is connected (or restored).
    """
    def __init__(self, u: AgentId, v: AgentId, params={}):
        super().__init__(u, v, params=params)

class RemoveLinkEvent(LinkUpdateEvent):
    """
    Event which router receives when link breakage is detected
    """
    def __init__(self, u: AgentId, v: AgentId):
        super().__init__(u, v)

##
# Conveyors events/actions
#

class Bag(Package):
    def __init__(self, bag_id, dst, start_time, contents):
        super().__init__(bag_id, 0, dst, start_time, contents)

class BagAppearanceEvent(WorldEvent):
    def __init__(self, src_id: int, bag: Bag):
        super().__init__(src_id=src_id, bag=bag)

class BagDetectionEvent(WorldEvent):
    def __init__(self, bag: Bag):
        super().__init__(bag=bag)

class BagReceiveAction(Action):
    def __init__(self, bag: Bag):
        super().__init__(bag=bag)

class DiverterKickAction(Action):
    def __init__(self):
        super().__init__()

class ConveyorSpeedChangeAction(Action):
    def __init__(self, new_speed: float):
        super().__init__(new_speed=new_speed)

#
# Service messages
#

class RewardMsg(ServiceMessage):
    def __init__(self, pkg_id: int, Q_estimate: float, reward_data):
        super().__init__(pkg_id=pkg_id, Q_estimate=Q_estimate, reward_data=reward_data)

class NetworkRewardMsg(RewardMsg):
    def __init__(self, pkg_id: int, Q_estimate: float, time_received: float):
        super().__init__(pkg_id, Q_estimate, time_received)

class ConveyorRewardMsg(RewardMsg):
    def __init__(self, bag_id: int, Q_estimate: float, time_processed: float,
                 energy_gap: float):
        super().__init__(bag_id, Q_estimate, (time_processed, energy_gap))

class StateAnnouncementMsg(ServiceMessage):
    def __init__(self, node: AgentId, seq: int, state):
        super().__init__(node=node, seq=seq, state=state)

class WrappedRouterMsg(ServiceMessage):
    """
    Wrapped message which allows to reuse router code in conveyors
    """
    def __init__(self, from_router: AgentId, to_router: AgentId, inner: Message):
        super().__init__(from_router=from_router, to_router=to_router, inner=inner)

#
# Conveyor control messages
#

class ConveyorBagMsg(ServiceMessage):
    def __init__(self, bag: Bag):
        super().__init__(bag=bag)

class IncomingBagMsg(ConveyorBagMsg):
    pass

class OutgoingBagMsg(ConveyorBagMsg):
    pass

class PassedBagMsg(ConveyorBagMsg):
    pass

class ConveyorStopMsg(ServiceMessage):
    pass

class StopTimeUpdMsg(ServiceMessage):
    def __init__(self, time: float):
        super().__init__(time=time)
