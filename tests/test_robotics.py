"""Tests for robotics controller and simulation adapter."""

from __future__ import annotations

import asyncio

import pytest

from sim.grocery_env import AISLE_POSITIONS, STORE_ITEMS, GroceryStoreEnv
from sous_bot.robotics.adapters.base import ActionStatus, RobotAction
from sous_bot.robotics.adapters.real import RealAdapter
from sous_bot.robotics.adapters.simulation import SimulationAdapter
from sous_bot.robotics.controller import RobotController, ShoppingItem


# --- GroceryStoreEnv tests ---


class TestGroceryStoreEnv:
    def test_load(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        assert env.model is not None
        assert env.data is not None
        assert env.model.nbody > 0

    def test_step(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        env.step(10)
        # Should not raise

    def test_robot_position(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        pos = env.get_robot_position()
        assert len(pos) == 3

    def test_item_positions(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        for item_name in STORE_ITEMS:
            pos = env.get_item_position(item_name)
            assert pos is not None, f"Item {item_name} not found"
            assert len(pos) == 3

    def test_item_info(self) -> None:
        info = GroceryStoreEnv().get_item_info("guanciale")
        assert info is not None
        assert info["aisle"] == "deli"

    def test_unknown_item(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        assert env.get_item_position("unicorn_meat") is None
        assert env.get_item_info("unicorn_meat") is None

    def test_mark_item_picked(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        assert env.mark_item_picked("guanciale")
        assert "guanciale" in env.items_in_cart
        assert "guanciale" not in env.available_items
        # Can't pick same item twice
        assert not env.mark_item_picked("guanciale")

    def test_reset(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        env.mark_item_picked("pasta")
        env.reset()
        assert "pasta" in env.available_items
        assert len(env.items_in_cart) == 0

    def test_render_frame(self) -> None:
        env = GroceryStoreEnv()
        env.load()
        frame = env.render_frame()
        assert frame is not None
        assert frame.shape[2] == 3  # RGB
        env.close()


# --- SimulationAdapter tests ---


class TestSimulationAdapter:
    @pytest.fixture
    def adapter(self) -> SimulationAdapter:
        adapter = SimulationAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.initialize())
        return adapter

    def test_initialize(self, adapter: SimulationAdapter) -> None:
        status = asyncio.get_event_loop().run_until_complete(adapter.status())
        assert status.is_ready
        assert status.adapter_type == "simulation"

    def test_locate_item(self, adapter: SimulationAdapter) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            adapter.locate_item("guanciale")
        )
        assert result.status == ActionStatus.COMPLETED
        assert result.data["aisle"] == "deli"

    def test_locate_unknown_item(self, adapter: SimulationAdapter) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            adapter.locate_item("unicorn_meat")
        )
        assert result.status == ActionStatus.FAILED

    def test_navigate(self, adapter: SimulationAdapter) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            adapter.navigate("dairy")
        )
        assert result.status == ActionStatus.COMPLETED

    def test_navigate_unknown(self, adapter: SimulationAdapter) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            adapter.navigate("narnia")
        )
        assert result.status == ActionStatus.FAILED

    def test_grasp_and_place(self, adapter: SimulationAdapter) -> None:
        loop = asyncio.get_event_loop()
        # Navigate to aisle
        loop.run_until_complete(adapter.navigate("deli"))
        # Reach
        loop.run_until_complete(adapter.reach("guanciale"))
        # Grasp
        result = loop.run_until_complete(adapter.grasp("guanciale"))
        assert result.status == ActionStatus.COMPLETED
        # Place in cart
        result = loop.run_until_complete(adapter.place_in_cart("guanciale"))
        assert result.status == ActionStatus.COMPLETED
        # Verify cart
        status = loop.run_until_complete(adapter.status())
        assert "guanciale" in status.items_in_cart

    def test_place_without_holding(self, adapter: SimulationAdapter) -> None:
        result = asyncio.get_event_loop().run_until_complete(
            adapter.place_in_cart("nothing")
        )
        assert result.status == ActionStatus.FAILED

    def test_execute_dispatch(self, adapter: SimulationAdapter) -> None:
        action = RobotAction(action="locate", target="pasta")
        result = asyncio.get_event_loop().run_until_complete(
            adapter.execute(action)
        )
        assert result.status == ActionStatus.COMPLETED

    def test_execute_unknown_action(self, adapter: SimulationAdapter) -> None:
        action = RobotAction(action="dance", target="floor")  # type: ignore[arg-type]
        result = asyncio.get_event_loop().run_until_complete(
            adapter.execute(action)
        )
        assert result.status == ActionStatus.FAILED


# --- RobotController tests ---


class TestRobotController:
    @pytest.fixture
    def controller(self) -> RobotController:
        adapter = SimulationAdapter()
        asyncio.get_event_loop().run_until_complete(adapter.initialize())
        return RobotController(adapter=adapter)

    def test_fetch_single_item(self, controller: RobotController) -> None:
        success = asyncio.get_event_loop().run_until_complete(
            controller.fetch_item("guanciale")
        )
        assert success

    def test_fetch_unknown_item(self, controller: RobotController) -> None:
        success = asyncio.get_event_loop().run_until_complete(
            controller.fetch_item("unicorn_meat")
        )
        assert not success

    def test_execute_shopping_list(self, controller: RobotController) -> None:
        items = [
            ShoppingItem(name="guanciale", quantity="150g", aisle="deli"),
            ShoppingItem(name="black pepper", quantity="1 tsp", aisle="spices"),
        ]
        result = asyncio.get_event_loop().run_until_complete(
            controller.execute_shopping_list(items)
        )
        assert result.completed
        assert "guanciale" in result.items_fetched
        assert "black pepper" in result.items_fetched
        assert len(result.items_failed) == 0

    def test_shopping_list_with_missing_item(self, controller: RobotController) -> None:
        items = [
            ShoppingItem(name="pasta"),
            ShoppingItem(name="unicorn_meat"),
        ]
        result = asyncio.get_event_loop().run_until_complete(
            controller.execute_shopping_list(items)
        )
        assert not result.completed
        assert "pasta" in result.items_fetched
        assert "unicorn_meat" in result.items_failed


# --- RealAdapter tests ---


class TestRealAdapter:
    def test_all_methods_fail(self) -> None:
        adapter = RealAdapter()
        loop = asyncio.get_event_loop()
        assert loop.run_until_complete(adapter.navigate("x")).status == ActionStatus.FAILED
        assert loop.run_until_complete(adapter.locate_item("x")).status == ActionStatus.FAILED
        assert loop.run_until_complete(adapter.reach("x")).status == ActionStatus.FAILED
        assert loop.run_until_complete(adapter.grasp("x")).status == ActionStatus.FAILED
        assert loop.run_until_complete(adapter.place_in_cart("x")).status == ActionStatus.FAILED
        assert loop.run_until_complete(adapter.hand_off("x")).status == ActionStatus.FAILED
        status = loop.run_until_complete(adapter.status())
        assert not status.is_ready
