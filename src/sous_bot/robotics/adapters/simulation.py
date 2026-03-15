"""MuJoCo simulation adapter for the grocery robot (real Unitree G1).

Pick-and-place motion pattern adapted from Robotics-Ark/ark_unitree_g1 (MIT License).
"""

from __future__ import annotations

import asyncio
import logging
import math

import numpy as np

from sim.grocery_env import GroceryStoreEnv, AISLE_POSITIONS, CART_POSITION

from .base import (
    ActionResult,
    ActionStatus,
    AdapterStatus,
    RobotAction,
    RobotAdapter,
)

logger = logging.getLogger(__name__)

# --- Timing constants ---
POSITION_TOLERANCE = 0.15
NAV_SPEED = 4.0
SIM_STEPS_PER_TICK = 50
FRAME_DELAY = 0.01

# --- Arm pose waypoints (joint-space, inspired by ark_unitree_g1 pick-and-place) ---
# Each pose is (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
ARM_REST = (0.3, -0.15, 0.0, 0.3, 0.0, 0.0, 0.0)
ARM_ABOVE = (-1.2, -0.4, 0.3, 0.8, 0.0, -0.3, 0.0)    # Arm raised, above shelf level
ARM_REACH = (-1.5, -0.5, 0.4, 1.2, 0.0, -0.5, 0.0)     # Arm extended forward to shelf
ARM_CARRY = (-0.8, -0.2, 0.1, 0.6, 0.0, -0.4, 0.0)     # Arm tucked holding item
ARM_DROP = (-1.0, -0.3, 0.0, 0.9, 0.0, -0.5, 0.0)      # Arm over cart to drop

# Hand grasp values per finger (from ark_unitree_g1 real robot values)
# Order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
HAND_OPEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HAND_CLOSED = [0.0, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8]


def _smooth(frac: float) -> float:
    """Smooth ease-in-out interpolation (Hermite)."""
    return frac * frac * (3.0 - 2.0 * frac)


class SimulationAdapter(RobotAdapter):
    """Robot adapter backed by MuJoCo simulation with real Unitree G1."""

    def __init__(self, env: GroceryStoreEnv | None = None) -> None:
        self.env = env or GroceryStoreEnv()
        self._current_action: RobotAction | None = None
        self._items_in_cart: list[str] = []
        self._held_item: str | None = None
        self._ready = False

    async def initialize(self) -> None:
        """Load the environment and mark adapter as ready."""
        self.env.load()
        self._ready = True
        logger.info("SimulationAdapter initialized with Unitree G1")

    async def navigate(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Navigate robot to target aisle or position."""
        action = RobotAction(action="navigate", target=target, parameters=parameters or {})
        self._current_action = action

        target_pos = self._resolve_target_position(target, parameters)
        if target_pos is None:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message=f"Unknown navigation target: {target}")

        await self._move_to(target_pos[0], target_pos[1])

        self._current_action = None
        robot_pos = self.env.get_robot_position()
        return ActionResult(
            action=action, status=ActionStatus.COMPLETED,
            message=f"Navigated to {target}",
            data={"position": robot_pos.tolist()},
        )

    async def locate_item(self, item_name: str, parameters: dict | None = None) -> ActionResult:
        """Locate an item in the store."""
        action = RobotAction(action="locate", target=item_name, parameters=parameters or {})
        self._current_action = action

        item_info = self.env.get_item_info(item_name)
        if item_info is None:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message=f"Item '{item_name}' not found in store")

        item_pos = self.env.get_item_position(item_name)
        self._current_action = None
        return ActionResult(
            action=action, status=ActionStatus.COMPLETED,
            message=f"Located {item_name} in {item_info['aisle']} aisle, shelf {item_info['shelf']}",
            data={"aisle": item_info["aisle"], "shelf": item_info["shelf"],
                  "position": item_pos.tolist() if item_pos is not None else None},
        )

    async def reach(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Reach arm toward an item using IK to target the exact item position."""
        action = RobotAction(action="reach", target=target, parameters=parameters or {})
        self._current_action = action

        item_pos = self.env.get_item_position(target)
        if item_pos is None:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message=f"Cannot see item '{target}' to reach for it")

        # Step 1: Raise arm above the item first
        logger.info("Raising arm to approach %s", target)
        above_pos = item_pos.copy()
        above_pos[2] += 0.15  # 15cm above item
        above_ik = self.env.solve_ik_right_arm(above_pos)
        if above_ik:
            await self._interpolate_arm(ARM_REST, above_ik, frames=20)
        else:
            # Fallback to fixed pose if IK fails
            robot_pos = self.env.get_robot_position()
            side = 1.0 if (item_pos[1] - robot_pos[1]) > 0 else -1.0
            above = list(ARM_ABOVE)
            above[1] *= side
            above[2] *= side
            await self._interpolate_arm(ARM_REST, tuple(above), frames=20)

        # Step 2: Lower arm to item position (IK to exact item location)
        logger.info("Reaching for %s", target)
        reach_ik = self.env.solve_ik_right_arm(item_pos)
        start_pose = above_ik if above_ik else ARM_ABOVE
        if reach_ik:
            await self._interpolate_arm(start_pose, reach_ik, frames=20)
        else:
            robot_pos = self.env.get_robot_position()
            side = 1.0 if (item_pos[1] - robot_pos[1]) > 0 else -1.0
            reach = list(ARM_REACH)
            reach[1] *= side
            reach[2] *= side
            await self._interpolate_arm(start_pose, tuple(reach), frames=20)

        # Step 3: Hold at item — hand should be right at the item
        await self._hold(10)

        # Save the reach pose for grasp to use when lifting
        self._last_reach_pose = reach_ik if reach_ik else ARM_REACH

        self._current_action = None
        return ActionResult(action=action, status=ActionStatus.COMPLETED,
                            message=f"Reached toward {target}")

    async def grasp(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Grasp an item — hand closes, item picked, arm lifts."""
        action = RobotAction(action="grasp", target=target, parameters=parameters or {})
        self._current_action = action

        if target not in self.env._items_on_shelf or not self.env._items_on_shelf[target]:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message=f"Cannot grasp '{target}' — not on shelf")

        # Step 1: Close hand around item (item stays on shelf visually)
        logger.info("Closing hand on %s", target)
        await self._interpolate_hand(HAND_OPEN, HAND_CLOSED, frames=20)

        # Step 2: Hold grip
        await self._hold(5)

        # Step 3: Item picked — instant snap to hand (no floating)
        self.env.mark_item_picked(target)

        # Step 4: Lift from shelf to carry position
        logger.info("Lifting %s from shelf", target)
        lift_from = getattr(self, '_last_reach_pose', ARM_REACH)
        await self._interpolate_arm(lift_from, ARM_CARRY, frames=25)

        self._held_item = target
        self._current_action = None
        logger.info("Grasped item: %s", target)
        return ActionResult(action=action, status=ActionStatus.COMPLETED,
                            message=f"Grasped {target}",
                            data={"held_item": target})

    async def place_in_cart(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Place the currently held item into the cart — full drop sequence."""
        action = RobotAction(action="place_in_cart", target=target, parameters=parameters or {})
        self._current_action = action

        if self._held_item is None:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message="Not holding any item to place in cart")

        item = self._held_item

        # Step 5: Lower arm over cart
        logger.info("Lowering %s over cart", item)
        await self._interpolate_arm(ARM_CARRY, ARM_DROP, frames=20)

        # Step 6: Open hand — item drops instantly into cart
        logger.info("Releasing %s into cart", item)
        await self._interpolate_hand(HAND_CLOSED, HAND_OPEN, frames=15)
        self.env.detach_item()

        # Step 7: Brief pause
        await self._hold(5)

        # Step 8: Return arm to rest
        logger.info("Returning arm to rest")
        await self._interpolate_arm(ARM_DROP, ARM_REST, frames=15)

        self._held_item = None
        self._items_in_cart.append(item)

        self._current_action = None
        logger.info("Placed %s in cart. Cart: %s", item, self._items_in_cart)
        return ActionResult(
            action=action, status=ActionStatus.COMPLETED,
            message=f"Placed {item} in cart",
            data={"items_in_cart": list(self._items_in_cart)},
        )

    async def hand_off(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Hand an item to the user."""
        action = RobotAction(action="hand_off", target=target, parameters=parameters or {})
        self._current_action = action

        if self._held_item is None and target in self._items_in_cart:
            self._items_in_cart.remove(target)
            self._held_item = target

        if self._held_item is None:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message=f"Item '{target}' not available for hand-off")

        item = self._held_item
        self._held_item = None
        self.env.detach_item()
        await self._interpolate_hand(HAND_CLOSED, HAND_OPEN, frames=8)
        await self._interpolate_arm(ARM_CARRY, ARM_REST, frames=12)

        self._current_action = None
        return ActionResult(action=action, status=ActionStatus.COMPLETED,
                            message=f"Handed off {item} to user")

    async def status(self) -> AdapterStatus:
        """Return current adapter status."""
        pos = self.env.get_robot_position() if self._ready else np.zeros(3)
        return AdapterStatus(
            adapter_type="simulation",
            is_ready=self._ready,
            robot_position=tuple(pos.tolist()),  # type: ignore[arg-type]
            current_action=self._current_action,
            items_in_cart=list(self._items_in_cart),
        )

    # --- Private animation helpers ---

    def _resolve_target_position(self, target: str, parameters: dict | None) -> tuple[float, float] | None:
        if parameters and "position" in parameters:
            p = parameters["position"]
            return (p[0], p[1])
        if target in AISLE_POSITIONS:
            return AISLE_POSITIONS[target]
        item_info = self.env.get_item_info(target)
        if item_info and item_info["aisle"] in AISLE_POSITIONS:
            return AISLE_POSITIONS[item_info["aisle"]]
        if target == "cart":
            return (CART_POSITION[0], CART_POSITION[1])
        return None

    async def _move_to(self, tx: float, ty: float) -> None:
        """Move robot toward target — teleport base with facing direction."""
        for _ in range(500):
            pos = self.env.get_robot_position()
            dx, dy = tx - pos[0], ty - pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < POSITION_TOLERANCE:
                self.env.set_robot_velocity(0, 0)
                self.env.step(SIM_STEPS_PER_TICK)
                return
            vx = NAV_SPEED * dx / dist
            vy = NAV_SPEED * dy / dist
            target_yaw = math.atan2(dy, dx)
            self.env.set_robot_heading(target_yaw)
            self.env.set_robot_velocity(vx, vy)
            self.env.step(SIM_STEPS_PER_TICK)
            await asyncio.sleep(FRAME_DELAY)

    async def _interpolate_arm(
        self, start: tuple, end: tuple, frames: int = 25
    ) -> None:
        """Smoothly interpolate right arm between two joint-space poses."""
        for t in range(frames):
            frac = _smooth((t + 1) / frames)
            pose = tuple(s * (1 - frac) + e * frac for s, e in zip(start, end))
            self.env.set_arm_targets(
                shoulder_pitch=pose[0],
                shoulder_roll=pose[1],
                shoulder_yaw=pose[2],
                elbow=pose[3],
                wrist_roll=pose[4],
                wrist_pitch=pose[5],
                wrist_yaw=pose[6],
                hand="right",
            )
            self.env.step(SIM_STEPS_PER_TICK)
            await asyncio.sleep(FRAME_DELAY)

    async def _interpolate_hand(
        self, start: list, end: list, frames: int = 15
    ) -> None:
        """Smoothly interpolate hand finger positions."""
        for t in range(frames):
            frac = _smooth((t + 1) / frames)
            values = [s * (1 - frac) + e * frac for s, e in zip(start, end)]
            self.env.set_hand_fingers(hand="right", values=values)
            self.env.step(SIM_STEPS_PER_TICK)
            await asyncio.sleep(FRAME_DELAY)

    async def _hold(self, frames: int = 10) -> None:
        """Hold current pose for a number of frames."""
        for _ in range(frames):
            self.env.step(SIM_STEPS_PER_TICK)
            await asyncio.sleep(FRAME_DELAY)
