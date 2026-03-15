"""Abstract base adapter for robot control."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RobotAction:
    """A single action for the robot to execute."""

    action: Literal["navigate", "locate", "reach", "grasp", "place_in_cart", "hand_off"]
    target: str
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_schema(cls, data: dict) -> "RobotAction":
        """Create from T1's schema dict (uses 'action_type' instead of 'action')."""
        action = data.get("action") or data.get("action_type", "")
        target = data.get("target", data.get("parameters", {}).get("item", ""))
        parameters = data.get("parameters", {})
        return cls(action=action, target=target, parameters=parameters)


@dataclass
class ActionResult:
    """Result of executing a robot action."""

    action: RobotAction
    status: ActionStatus
    message: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class AdapterStatus:
    """Current status of the robot adapter."""

    adapter_type: str
    is_ready: bool
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_action: RobotAction | None = None
    items_in_cart: list[str] = field(default_factory=list)


class RobotAdapter(ABC):
    """Abstract base class for robot adapters (simulation or real hardware)."""

    @abstractmethod
    async def navigate(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Navigate the robot to a target location (e.g., aisle, shelf)."""

    @abstractmethod
    async def locate_item(self, item_name: str, parameters: dict | None = None) -> ActionResult:
        """Locate an item on the shelves."""

    @abstractmethod
    async def reach(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Extend arm to reach for an item."""

    @abstractmethod
    async def grasp(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Grasp an item."""

    @abstractmethod
    async def place_in_cart(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Place a grasped item into the cart."""

    @abstractmethod
    async def hand_off(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Hand an item off to the user."""

    @abstractmethod
    async def status(self) -> AdapterStatus:
        """Return the current status of the adapter."""

    async def execute(self, action: RobotAction) -> ActionResult:
        """Dispatch a RobotAction to the appropriate method."""
        dispatch = {
            "navigate": self.navigate,
            "locate": self.locate_item,
            "reach": self.reach,
            "grasp": self.grasp,
            "place_in_cart": self.place_in_cart,
            "hand_off": self.hand_off,
        }
        handler = dispatch.get(action.action)
        if handler is None:
            return ActionResult(
                action=action,
                status=ActionStatus.FAILED,
                message=f"Unknown action: {action.action}",
            )
        return await handler(action.target, action.parameters)
