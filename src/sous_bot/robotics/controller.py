"""Robot controller orchestrator — takes a shopping list and executes the fetch sequence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sim.grocery_env import AISLE_POSITIONS, STORE_ITEMS

from .adapters.base import ActionResult, ActionStatus, RobotAction, RobotAdapter

logger = logging.getLogger(__name__)


@dataclass
class ShoppingItem:
    """An item to fetch from the store."""

    name: str
    quantity: str = "1"
    aisle: str | None = None


@dataclass
class FetchSequenceResult:
    """Result of executing a full shopping fetch sequence."""

    items_fetched: list[str] = field(default_factory=list)
    items_failed: list[str] = field(default_factory=list)
    action_log: list[ActionResult] = field(default_factory=list)
    completed: bool = False


class RobotController:
    """Orchestrates shopping list execution through the robot adapter."""

    def __init__(self, adapter: RobotAdapter) -> None:
        self.adapter = adapter
        self._action_log: list[ActionResult] = []

    async def fetch_item(self, item_name: str) -> bool:
        """Execute the full fetch sequence for a single item.

        Sequence: locate → navigate to item → reach → grasp → navigate to cart → place_in_cart
        """
        # 1. Locate the item
        result = await self.adapter.execute(
            RobotAction(action="locate", target=item_name)
        )
        self._action_log.append(result)
        if result.status != ActionStatus.COMPLETED:
            logger.warning("Failed to locate %s: %s", item_name, result.message)
            return False

        aisle = result.data.get("aisle", "")
        item_pos = result.data.get("position")
        vision_detected = result.data.get("vision_detected", False)
        logger.info("Item %s located in %s aisle at %s (vision=%s)",
                     item_name, aisle, item_pos, vision_detected)

        # 2. Navigate to the item
        if vision_detected and item_pos:
            # Vision already walked us to the aisle during scanning —
            # just stand near the detected position
            stand_y = item_pos[1] * 0.6
            result = await self.adapter.execute(
                RobotAction(action="navigate", target=item_name,
                            parameters={"position": [item_pos[0], stand_y]})
            )
        elif item_pos:
            stand_y = item_pos[1] * 0.6
            result = await self.adapter.execute(
                RobotAction(action="navigate", target=item_name,
                            parameters={"position": [item_pos[0], stand_y]})
            )
        else:
            result = await self.adapter.execute(
                RobotAction(action="navigate", target=aisle)
            )
        self._action_log.append(result)
        if result.status != ActionStatus.COMPLETED:
            logger.warning("Failed to navigate to %s: %s", item_name, result.message)
            return False

        # 3. Reach for the item
        result = await self.adapter.execute(
            RobotAction(action="reach", target=item_name)
        )
        self._action_log.append(result)
        if result.status != ActionStatus.COMPLETED:
            logger.warning("Failed to reach %s: %s", item_name, result.message)
            return False

        # 4. Grasp the item
        result = await self.adapter.execute(
            RobotAction(action="grasp", target=item_name)
        )
        self._action_log.append(result)
        if result.status != ActionStatus.COMPLETED:
            logger.warning("Failed to grasp %s: %s", item_name, result.message)
            return False

        # 5. Navigate back to the cart
        result = await self.adapter.execute(
            RobotAction(action="navigate", target="cart",
                        parameters={"position": [0.5, -0.3]})
        )
        self._action_log.append(result)

        # 6. Place in cart
        result = await self.adapter.execute(
            RobotAction(action="place_in_cart", target=item_name)
        )
        self._action_log.append(result)
        if result.status != ActionStatus.COMPLETED:
            logger.warning("Failed to place %s in cart: %s", item_name, result.message)
            return False

        logger.info("Successfully fetched %s", item_name)
        return True

    def _plan_route(self, items: list) -> list:
        """Sort items by aisle position for an efficient route through the store."""
        aisle_order = {name: i for i, name in enumerate(AISLE_POSITIONS.keys())}

        def sort_key(item: ShoppingItem) -> int:
            aisle = item.aisle
            if aisle is None:
                store_info = STORE_ITEMS.get(item.name)
                aisle = store_info["aisle"] if store_info else None
            return aisle_order.get(aisle, 999) if aisle else 999

        return sorted(items, key=sort_key)

    async def execute_shopping_list(self, items: list) -> FetchSequenceResult:
        """Execute a full grocery shopping sequence.

        Accepts T2 ShoppingItem dataclasses or T1 Pydantic ShoppingItem — any object with .name works.
        """
        result = FetchSequenceResult()
        sorted_items = self._plan_route(items)

        logger.info("Starting shopping run for %d items", len(sorted_items))
        for item in sorted_items:
            logger.info("Fetching: %s", item.name)
            success = await self.fetch_item(item.name)
            if success:
                result.items_fetched.append(item.name)
            else:
                result.items_failed.append(item.name)

        # Already at cart from last item placement

        result.action_log = list(self._action_log)
        result.completed = len(result.items_failed) == 0
        self._action_log.clear()

        logger.info("Shopping complete. Fetched: %s, Failed: %s",
                     result.items_fetched, result.items_failed)
        return result
