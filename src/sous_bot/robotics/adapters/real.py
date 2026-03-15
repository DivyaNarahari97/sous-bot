"""Real Unitree G1 hardware adapter (stretch goal stub)."""

from __future__ import annotations

from .base import ActionResult, ActionStatus, AdapterStatus, RobotAction, RobotAdapter


class RealAdapter(RobotAdapter):
    """Stub adapter for real Unitree G1 hardware.

    Not yet implemented — requires hardware SDK integration and safety review.
    """

    async def navigate(self, target: str, parameters: dict | None = None) -> ActionResult:
        return ActionResult(
            action=RobotAction(action="navigate", target=target, parameters=parameters or {}),
            status=ActionStatus.FAILED,
            message="RealAdapter not implemented — use SimulationAdapter",
        )

    async def locate_item(self, item_name: str, parameters: dict | None = None) -> ActionResult:
        return ActionResult(
            action=RobotAction(action="locate", target=item_name, parameters=parameters or {}),
            status=ActionStatus.FAILED,
            message="RealAdapter not implemented",
        )

    async def reach(self, target: str, parameters: dict | None = None) -> ActionResult:
        return ActionResult(
            action=RobotAction(action="reach", target=target, parameters=parameters or {}),
            status=ActionStatus.FAILED,
            message="RealAdapter not implemented",
        )

    async def grasp(self, target: str, parameters: dict | None = None) -> ActionResult:
        return ActionResult(
            action=RobotAction(action="grasp", target=target, parameters=parameters or {}),
            status=ActionStatus.FAILED,
            message="RealAdapter not implemented",
        )

    async def place_in_cart(self, target: str, parameters: dict | None = None) -> ActionResult:
        return ActionResult(
            action=RobotAction(action="place_in_cart", target=target, parameters=parameters or {}),
            status=ActionStatus.FAILED,
            message="RealAdapter not implemented",
        )

    async def hand_off(self, target: str, parameters: dict | None = None) -> ActionResult:
        return ActionResult(
            action=RobotAction(action="hand_off", target=target, parameters=parameters or {}),
            status=ActionStatus.FAILED,
            message="RealAdapter not implemented",
        )

    async def status(self) -> AdapterStatus:
        return AdapterStatus(adapter_type="real", is_ready=False)
