#!/usr/bin/env python3
"""PantryPilot demo — full grocery shopping sequence in MuJoCo simulation.

Usage:
    python scripts/run_demo.py              # headless (logs only)
    python scripts/run_demo.py --viewer     # interactive 3D viewer (for live demo)
    python scripts/run_demo.py --record     # save video to demo_output.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from sim.grocery_env import GroceryStoreEnv
from sous_bot.robotics.adapters.simulation import SimulationAdapter
from sous_bot.robotics.controller import RobotController, ShoppingItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("demo")

# Sample shopping list — what the planner would generate for carbonara
DEMO_SHOPPING_LIST = [
    ShoppingItem(name="guanciale", quantity="150g", aisle="deli"),
    ShoppingItem(name="black pepper", quantity="1 tsp", aisle="spices"),
    ShoppingItem(name="olive oil", quantity="1 bottle", aisle="produce"),
]


async def run_demo_headless(
    shopping_list: list[ShoppingItem] | None = None,
) -> object:
    """Run the full grocery shopping demo (headless)."""
    items = shopping_list or DEMO_SHOPPING_LIST

    logger.info("=" * 60)
    logger.info("PantryPilot — Grocery Shopping Demo")
    logger.info("=" * 60)
    logger.info("Shopping list: %s", [item.name for item in items])

    env = GroceryStoreEnv()
    adapter = SimulationAdapter(env=env)
    await adapter.initialize()

    controller = RobotController(adapter=adapter)

    status = await adapter.status()
    logger.info("Robot status: ready=%s, position=%s", status.is_ready, status.robot_position)

    logger.info("-" * 60)
    logger.info("Starting shopping run...")
    logger.info("-" * 60)

    result = await controller.execute_shopping_list(items)

    logger.info("=" * 60)
    logger.info("Shopping Run Complete!")
    logger.info("  Items fetched: %s", result.items_fetched)
    logger.info("  Items failed:  %s", result.items_failed)
    logger.info("  Total actions: %d", len(result.action_log))
    logger.info("  Success:       %s", result.completed)
    logger.info("=" * 60)

    final_status = await adapter.status()
    logger.info("Cart contents: %s", final_status.items_in_cart)

    env.close()
    return result


def run_demo_with_viewer(shopping_list: list[ShoppingItem] | None = None) -> None:
    """Run the demo with MuJoCo's interactive 3D viewer for live judging."""
    items = shopping_list or DEMO_SHOPPING_LIST

    logger.info("=" * 60)
    logger.info("PantryPilot — Interactive Grocery Shopping Demo")
    logger.info("=" * 60)
    logger.info("Shopping list: %s", [item.name for item in items])
    logger.info("Controls: drag to rotate, scroll to zoom, double-click to track body")

    env = GroceryStoreEnv()
    env.load()

    adapter = SimulationAdapter(env=env)
    adapter._ready = True
    controller = RobotController(adapter=adapter)

    shopping_done = threading.Event()
    shopping_result = [None]

    async def _run_shopping() -> None:
        await asyncio.sleep(1.0)
        logger.info("Starting shopping run...")
        result = await controller.execute_shopping_list(items)
        shopping_result[0] = result

        logger.info("=" * 60)
        logger.info("Shopping Run Complete!")
        logger.info("  Items fetched: %s", result.items_fetched)
        logger.info("  Items failed:  %s", result.items_failed)
        logger.info("  Success:       %s", result.completed)
        logger.info("=" * 60)

        final_status = await adapter.status()
        logger.info("Cart contents: %s", final_status.items_in_cart)
        logger.info("Close the viewer window to exit.")

    def shopping_thread() -> None:
        asyncio.run(_run_shopping())
        shopping_done.set()

    t = threading.Thread(target=shopping_thread, daemon=True)
    t.start()

    mujoco.viewer.launch(env.model, env.data)
    env.close()


def run_demo_with_passive_viewer(
    shopping_list: list[ShoppingItem] | None = None,
) -> None:
    """Run demo with passive viewer — shopping drives the sim, viewer just watches."""
    items = shopping_list or DEMO_SHOPPING_LIST

    logger.info("=" * 60)
    logger.info("PantryPilot — Passive Viewer Demo")
    logger.info("=" * 60)
    logger.info("Shopping list: %s", [item.name for item in items])

    env = GroceryStoreEnv()
    env.load()

    adapter = SimulationAdapter(env=env)
    adapter._ready = True
    controller = RobotController(adapter=adapter)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 8.0
        viewer.cam.lookat[:] = [3.0, 0.0, 1.0]

        async def _run_shopping_with_sync() -> object:
            await asyncio.sleep(0.5)
            logger.info("Starting shopping run...")
            result = await controller.execute_shopping_list(items)
            logger.info("=" * 60)
            logger.info("Shopping Complete! Fetched: %s", result.items_fetched)
            logger.info("=" * 60)
            return result

        original_step = env.step

        def step_with_viewer_sync(n_steps: int = 1) -> None:
            original_step(n_steps)
            viewer.sync()

        env.step = step_with_viewer_sync  # type: ignore[method-assign]

        result = asyncio.run(_run_shopping_with_sync())

        logger.info("Shopping done. Viewer stays open — close window to exit.")
        while viewer.is_running():
            time.sleep(0.1)

    env.close()


def run_demo_record(
    shopping_list: list[ShoppingItem] | None = None,
    output_path: str = "demo_output.mp4",
    fps: int = 30,
) -> None:
    """Record the demo to an MP4 video file."""
    items = shopping_list or DEMO_SHOPPING_LIST

    logger.info("=" * 60)
    logger.info("PantryPilot — Recording Demo to %s", output_path)
    logger.info("=" * 60)

    env = GroceryStoreEnv()
    env.load()

    adapter = SimulationAdapter(env=env)
    adapter._ready = True
    controller = RobotController(adapter=adapter)

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=720, width=1280)

    original_step = env.step

    def step_and_capture(n_steps: int = 1) -> None:
        original_step(n_steps)
        renderer.update_scene(env.data)
        frames.append(renderer.render().copy())

    env.step = step_and_capture  # type: ignore[method-assign]

    result = asyncio.run(_run_shopping_for_record(controller, items))

    renderer.close()

    if frames:
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            logger.info("Saved %d frames to %s (%dx%d @ %d fps)",
                         len(frames), output_path, w, h, fps)
        except ImportError:
            npy_path = output_path.replace(".mp4", "_frames.npy")
            np.save(npy_path, np.stack(frames))
            logger.info("cv2 not available. Saved %d frames to %s", len(frames), npy_path)

    env.close()
    logger.info("Items fetched: %s", result.items_fetched)


async def _run_shopping_for_record(
    controller: RobotController,
    items: list[ShoppingItem],
) -> object:
    await asyncio.sleep(0.01)
    return await controller.execute_shopping_list(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="PantryPilot Grocery Shopping Demo")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--viewer", action="store_true",
                      help="Launch interactive 3D viewer (for live demo)")
    mode.add_argument("--passive", action="store_true",
                      help="Launch passive viewer with auto camera")
    mode.add_argument("--record", action="store_true",
                      help="Record demo to MP4 video")
    parser.add_argument("--output", default="demo_output.mp4",
                        help="Output video path (with --record)")
    parser.add_argument("--items", nargs="+", default=None,
                        help="Custom item names to fetch (e.g., --items pasta eggs milk)")
    args = parser.parse_args()

    shopping_list = None
    if args.items:
        shopping_list = [ShoppingItem(name=name) for name in args.items]

    if args.viewer:
        run_demo_with_viewer(shopping_list)
    elif args.passive:
        run_demo_with_passive_viewer(shopping_list)
    elif args.record:
        run_demo_record(shopping_list, output_path=args.output)
    else:
        result = asyncio.run(run_demo_headless(shopping_list))
        sys.exit(0 if result.completed else 1)


if __name__ == "__main__":
    main()
