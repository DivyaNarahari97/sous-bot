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
NAV_SPEED = 2.5          # Slower, more realistic walking
SIM_STEPS_PER_TICK = 50
FRAME_DELAY = 0.02       # Slower tick — more visible motion

# Cart bounds for avoidance (cart is at ~0.5, -0.3)
CART_CENTER = (0.5, -0.3)
CART_AVOID_RADIUS = 0.6  # Stay this far from cart center when navigating

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

    def __init__(self, env: GroceryStoreEnv | None = None, use_vision: bool = True) -> None:
        self.env = env or GroceryStoreEnv()
        self._current_action: RobotAction | None = None
        self._items_in_cart: list[str] = []
        self._held_item: str | None = None
        self._ready = False
        self._use_vision = use_vision
        self._detector = None  # Lazy-loaded VLM detector

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
        """Locate an item in the store (lookup only — VLM scan happens at shelf)."""
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
            message=f"Located {item_name} in {item_info['aisle']} aisle",
            data={"aisle": item_info["aisle"], "shelf": item_info["shelf"],
                  "position": item_pos.tolist() if item_pos is not None else None},
        )

    async def _vision_locate(self, item_name: str) -> bool:
        """Use Qwen2.5-VL to visually confirm an item on the shelf."""
        try:
            # Lazy-load the detector
            if self._detector is None:
                from sous_bot.vision.detector import IngredientDetector
                self._detector = IngredientDetector()
                logger.info("VLM vision detector initialized (Qwen2.5-VL)")

            # Capture what the robot sees
            image_bytes = self.env.render_robot_view_jpeg()
            if image_bytes is None:
                logger.warning("Could not capture robot view")
                return False

            # Ask VLM: "Do you see this item on the shelf?"
            logger.info("VLM scanning shelf for '%s'...", item_name)
            result = await asyncio.to_thread(
                self._detector.locate_item_on_shelf, image_bytes, item_name
            )

            if result.found and result.confidence >= 0.5:
                logger.info(
                    "VLM found '%s' at %s (confidence: %.2f) — %s",
                    item_name, result.position, result.confidence, result.description,
                )
                return True
            else:
                logger.info(
                    "VLM did not confidently find '%s' (found=%s, conf=%.2f)",
                    item_name, result.found, result.confidence,
                )
                return False
        except Exception as e:
            logger.warning("VLM vision failed: %s — proceeding without vision", e)
            return False

    async def reach(self, target: str, parameters: dict | None = None) -> ActionResult:
        """Reach arm toward an item using IK to target the exact item position."""
        action = RobotAction(action="reach", target=target, parameters=parameters or {})
        self._current_action = action

        # VLM scan: robot is at the shelf now — use vision to confirm item
        if self._use_vision:
            found = await self._vision_locate(target)
            if found:
                logger.info("VLM confirmed '%s' on shelf — reaching", target)
            else:
                logger.info("VLM did not confirm '%s' — reaching by store lookup", target)

        item_pos = self.env.get_item_position(target)
        if item_pos is None:
            return ActionResult(action=action, status=ActionStatus.FAILED,
                                message=f"Cannot see item '{target}' to reach for it")

        # Step 1: Open hand first (prepare to grab)
        logger.info("Opening hand, preparing to reach for %s", target)
        await self._interpolate_hand(HAND_CLOSED, HAND_OPEN, frames=15)
        await self._hold(8)

        # Step 2: Raise arm above the item first
        logger.info("Raising arm to approach %s", target)
        above_pos = item_pos.copy()
        above_pos[2] += 0.15  # 15cm above item
        above_ik = self.env.solve_ik_right_arm(above_pos)
        if above_ik:
            await self._interpolate_arm(ARM_REST, above_ik, frames=40)
        else:
            robot_pos = self.env.get_robot_position()
            side = 1.0 if (item_pos[1] - robot_pos[1]) > 0 else -1.0
            above = list(ARM_ABOVE)
            above[1] *= side
            above[2] *= side
            await self._interpolate_arm(ARM_REST, tuple(above), frames=40)

        # Step 3: Pause above — looking at item
        await self._hold(15)

        # Step 4: Lower arm carefully to item position
        logger.info("Reaching for %s", target)
        reach_ik = self.env.solve_ik_right_arm(item_pos)
        start_pose = above_ik if above_ik else ARM_ABOVE
        if reach_ik:
            await self._interpolate_arm(start_pose, reach_ik, frames=35)
        else:
            robot_pos = self.env.get_robot_position()
            side = 1.0 if (item_pos[1] - robot_pos[1]) > 0 else -1.0
            reach = list(ARM_REACH)
            reach[1] *= side
            reach[2] *= side
            await self._interpolate_arm(start_pose, tuple(reach), frames=35)

        # Step 5: Hold at item — positioning hand
        await self._hold(15)

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

        # Step 1: Close hand slowly around item
        logger.info("Closing hand on %s", target)
        await self._interpolate_hand(HAND_OPEN, HAND_CLOSED, frames=35)

        # Step 2: Hold grip — securing item
        logger.info("Securing grip on %s", target)
        await self._hold(15)

        # Step 3: Item picked — snap to hand
        self.env.mark_item_picked(target)

        # Step 4: Slight pull back before lifting (tug off shelf)
        lift_from = getattr(self, '_last_reach_pose', ARM_REACH)
        tug_pose = tuple(
            lf * 0.85 + carry * 0.15 for lf, carry in zip(lift_from, ARM_CARRY)
        )
        logger.info("Pulling %s from shelf", target)
        await self._interpolate_arm(lift_from, tug_pose, frames=20)
        await self._hold(10)

        # Step 5: Lift from shelf to carry position
        logger.info("Lifting %s from shelf", target)
        await self._interpolate_arm(tug_pose, ARM_CARRY, frames=40)

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

        # Step 5: Extend arm outward toward cart
        logger.info("Extending arm over cart with %s", item)
        await self._interpolate_arm(ARM_CARRY, ARM_DROP, frames=40)

        # Step 6: Pause — positioning over cart
        await self._hold(15)

        # Step 7: Open hand slowly — item drops into cart
        logger.info("Releasing %s into cart", item)
        await self._interpolate_hand(HAND_CLOSED, HAND_OPEN, frames=30)
        self.env.detach_item()

        # Step 8: Pause — watch item settle
        await self._hold(15)

        # Step 9: Return arm to rest slowly
        logger.info("Returning arm to rest")
        await self._interpolate_arm(ARM_DROP, ARM_REST, frames=30)

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

    def _plan_waypoints(self, tx: float, ty: float) -> list[tuple[float, float]]:
        """Plan waypoints that avoid the cart."""
        pos = self.env.get_robot_position()
        sx, sy = pos[0], pos[1]

        # Check if direct path passes near cart
        cx, cy = CART_CENTER
        # Simple check: if start or end is near cart, add a detour waypoint
        steps = 10
        needs_detour = False
        for i in range(steps + 1):
            t = i / steps
            px = sx + t * (tx - sx)
            py = sy + t * (ty - sy)
            if math.sqrt((px - cx) ** 2 + (py - cy) ** 2) < CART_AVOID_RADIUS:
                needs_detour = True
                break

        if needs_detour:
            # Go around the cart on the positive-y side
            detour_y = cy + CART_AVOID_RADIUS + 0.3
            waypoints = []
            # First go to detour y at current x
            if abs(sy - cy) < CART_AVOID_RADIUS + 0.2:
                waypoints.append((sx, detour_y))
            # Then traverse past cart
            waypoints.append((tx, detour_y))
            # Then go to target
            waypoints.append((tx, ty))
            return waypoints

        return [(tx, ty)]

    async def _move_to(self, tx: float, ty: float) -> None:
        """Move robot toward target via waypoints, avoiding the cart."""
        waypoints = self._plan_waypoints(tx, ty)
        for wx, wy in waypoints:
            await self._move_to_point(wx, wy)

    async def _move_to_point(self, tx: float, ty: float) -> None:
        """Move robot toward a single point."""
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
