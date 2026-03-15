#!/usr/bin/env python3
"""PantryPilot — Full integrated pipeline.

Flow:
    1. User enters recipes they want to cook
    2. API calls Nebius LLM to extract all required ingredients
    3. GET /generate-shopping-list returns matched store items
    4. Robot simulation picks all matched items

Usage:
    uv run python scripts/run_viewer.py
    uv run python scripts/run_viewer.py --items carbonara "stir fry"

Controls:
    Left-click + drag  : Rotate camera
    Right-click + drag : Pan camera
    Scroll             : Zoom in/out
    Space              : Start/restart shopping sequence
    R                  : Reset environment
    Q / ESC            : Quit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import threading
import time

import glfw
import mujoco
import numpy as np
import requests as http_requests
import uvicorn
from dotenv import load_dotenv

from sim.grocery_env import GroceryStoreEnv, STORE_ITEMS
from sous_bot.robotics.adapters.simulation import SimulationAdapter
from sous_bot.robotics.controller import RobotController, ShoppingItem

load_dotenv()

API_PORT = 8321  # Local port for T1's API server

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("pantrypilot")

WIDTH, HEIGHT = 1280, 720


# ── T1 API Server ────────────────────────────────────────────────────────────

def start_api_server() -> None:
    """Start T1's FastAPI server in a background thread."""
    from sous_bot.api.main import app

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Wait for server to be ready
    for _ in range(30):
        try:
            http_requests.get(f"http://127.0.0.1:{API_PORT}/docs", timeout=1)
            logger.info("T1 API server ready on port %d", API_PORT)
            return
        except Exception:
            time.sleep(0.5)
    logger.warning("T1 API server may not be ready — continuing anyway")


# ── Phase 1+2: Recipe → Shopping List (via T1 /chat + /shopping-list) ────────

def get_shopping_list_from_api(recipes: list[str]) -> list[ShoppingItem]:
    """Call T1's /chat endpoint to get a shopping list for recipes."""
    base = f"http://127.0.0.1:{API_PORT}"
    store_item_names = sorted(STORE_ITEMS.keys())

    message = (
        f"I want to cook {', '.join(recipes)} for 1 day, 2 servings per day. "
        f"Give me the complete shopping list immediately. "
        f"Only use items from this store: {', '.join(store_item_names)}"
    )

    logger.info("Calling T1 /chat API for recipes: %s", recipes)
    try:
        resp = http_requests.post(
            f"{base}/chat",
            json={"message": message, "available_ingredients": []},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = data.get("session_id", "")
        logger.info("T1 /chat response: %s", data.get("message", "")[:100])
    except Exception as e:
        logger.error("T1 /chat failed: %s", e)
        return []

    # Try to get structured shopping list
    try:
        resp2 = http_requests.get(
            f"{base}/generate-shopping-list",
            params={"session_id": session_id},
            timeout=10,
        )
        if resp2.ok:
            sl_data = resp2.json()
            items = []
            store_names = set(STORE_ITEMS.keys())
            for recipe_block in sl_data.get("recipes", []):
                for item in recipe_block.get("items", []):
                    name = item.get("name", "").lower().strip()
                    # Match to store
                    if name in store_names:
                        info = STORE_ITEMS[name]
                        items.append(ShoppingItem(
                            name=name,
                            quantity=item.get("quantity", "1"),
                            aisle=info["aisle"],
                        ))
                    else:
                        for store_name in store_names:
                            if name in store_name or store_name in name:
                                info = STORE_ITEMS[store_name]
                                items.append(ShoppingItem(
                                    name=store_name,
                                    quantity=item.get("quantity", "1"),
                                    aisle=info["aisle"],
                                ))
                                break

            # Deduplicate
            seen = set()
            unique = []
            for it in items:
                if it.name not in seen:
                    seen.add(it.name)
                    unique.append(it)
            if unique:
                logger.info("T1 API returned %d matched store items", len(unique))
                return unique
            logger.warning("T1 /shopping-list returned no matchable items")
    except Exception as e:
        logger.warning("T1 /shopping-list failed: %s — will parse chat reply", e)

    return []


# ── Phase 3: Interactive Viewer + Simulation ─────────────────────────────────

class GroceryViewer:
    """Interactive GLFW viewer for the grocery sim."""

    def __init__(self, env: GroceryStoreEnv, items: list[ShoppingItem], use_vision: bool = True) -> None:
        self.env = env
        self.items = items
        self._use_vision = use_vision

        # MuJoCo rendering objects
        self.scene = mujoco.MjvScene(env.model, maxgeom=5000)
        self.context: mujoco.MjrContext | None = None
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.viewport = mujoco.MjrRect(0, 0, WIDTH, HEIGHT)

        # Camera defaults
        self.camera.azimuth = 145
        self.camera.elevation = -25
        self.camera.distance = 8.0
        self.camera.lookat[:] = [3.0, 0.0, 1.0]

        # Mouse state
        self._button_left = False
        self._button_right = False
        self._button_middle = False
        self._last_x = 0.0
        self._last_y = 0.0

        # Shopping state
        self._shopping_running = False
        self._shopping_done = False
        self._status_text = "Press SPACE to start shopping"
        self._fetched: list[str] = []
        self._failed: list[str] = []
        self._current_item: str = ""
        self._current_step: str = ""
        self._log_lines: list[str] = []
        self._pick_notification: str = ""
        self._pick_notification_time: float = 0.0

        # Robot camera (PiP)
        self._robot_camera = mujoco.MjvCamera()
        self._robot_scene = mujoco.MjvScene(env.model, maxgeom=5000)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._last_x, self._last_y = glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window, xpos, ypos):
        dx = xpos - self._last_x
        dy = ypos - self._last_y
        self._last_x = xpos
        self._last_y = ypos
        width, height = glfw.get_window_size(window)
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        if self._button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            return
        mujoco.mjv_moveCamera(self.env.model, action, dx / width, dy / height,
                              self.scene, self.camera)

    def _scroll_callback(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.env.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0, -0.15 * yoffset, self.scene, self.camera)

    def _key_callback(self, window, key, scancode, act, mods):
        if act != glfw.PRESS:
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            if not self._shopping_running:
                if self._shopping_done:
                    self.env.reset()
                    self._shopping_done = False
                self._start_shopping()
        elif key == glfw.KEY_R:
            self.env.reset()
            self._shopping_done = False
            self._shopping_running = False
            self._status_text = "Environment reset. Press SPACE to start."
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            self.camera.distance = max(1.0, self.camera.distance - 1.0)
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            self.camera.distance = min(20.0, self.camera.distance + 1.0)
        elif key == glfw.KEY_UP:
            self.camera.elevation = min(0, self.camera.elevation + 5)
        elif key == glfw.KEY_DOWN:
            self.camera.elevation = max(-90, self.camera.elevation - 5)
        elif key == glfw.KEY_LEFT:
            self.camera.azimuth -= 10
        elif key == glfw.KEY_RIGHT:
            self.camera.azimuth += 10

    def _log(self, msg: str):
        """Add a line to the on-screen log (keep last 8 lines)."""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > 8:
            self._log_lines = self._log_lines[-8:]
        logger.info(msg)

    def _show_pick(self, item_name: str):
        """Flash a pick notification on screen."""
        self._pick_notification = f"PICKED: {item_name.upper()}"
        self._pick_notification_time = time.time()

    def _start_shopping(self):
        self._shopping_running = True
        self._shopping_done = False
        self._log_lines = []
        self._log(f"Starting shopping run for {len(self.items)} items")

        def _run():
            adapter = SimulationAdapter(env=self.env, use_vision=self._use_vision)
            adapter._ready = True
            controller = RobotController(adapter=adapter)

            fetched = []
            failed = []
            for i, item in enumerate(self.items):
                self._current_item = item.name
                aisle = item.aisle or "unknown"

                self._current_step = "Locating"
                self._log(f"Looking for {item.name} in {aisle} aisle")
                self._status_text = f"[{i+1}/{len(self.items)}] Locating {item.name}..."

                self._current_step = "Navigating"
                self._status_text = f"[{i+1}/{len(self.items)}] Walking to {item.name}..."

                self._current_step = "Reaching"

                self._current_step = "Picking"
                success = asyncio.run(controller.fetch_item(item.name))

                if success:
                    fetched.append(item.name)
                    self._fetched = list(fetched)  # Update HUD immediately
                    self._show_pick(item.name)
                    self._log(f"Picked {item.name} -> cart ({len(fetched)}/{len(self.items)})")
                else:
                    failed.append(item.name)
                    self._failed = list(failed)
                    self._log(f"FAILED to get {item.name}")

                self._status_text = f"Cart: {len(fetched)}/{len(self.items)} items"
            self._current_item = ""
            self._current_step = ""

            if failed:
                self._log(f"Done! Got {len(fetched)}, missed {len(failed)}: {failed}")
            else:
                self._log(f"Done! All {len(fetched)} items collected!")

            self._status_text = f"Shopping complete! {len(fetched)}/{len(self.items)} items. Press R to reset."
            self._shopping_done = True
            self._shopping_running = False

        threading.Thread(target=_run, daemon=True).start()

    def run(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        window = glfw.create_window(WIDTH, HEIGHT, "PantryPilot — Grocery Shopping Robot", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        glfw.set_mouse_button_callback(window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(window, self._mouse_move_callback)
        glfw.set_scroll_callback(window, self._scroll_callback)
        glfw.set_key_callback(window, self._key_callback)

        self.context = mujoco.MjrContext(self.env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        logger.info("Viewer ready. Press SPACE to start shopping, Q to quit.")

        # Auto-start shopping after 1 second
        auto_start_time = time.time() + 1.0
        auto_started = False

        while not glfw.window_should_close(window):
            if not auto_started and time.time() > auto_start_time:
                auto_started = True
                self._start_shopping()

            if not self._shopping_running:
                self.env.step(1)

            fb_width, fb_height = glfw.get_framebuffer_size(window)
            self.viewport.width = fb_width
            self.viewport.height = fb_height

            mujoco.mjv_updateScene(
                self.env.model, self.env.data, self.option, None,
                self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene,
            )
            mujoco.mjr_render(self.viewport, self.scene, self.context)

            # ── Robot camera PiP (bottom-right) ──
            pip_w, pip_h = fb_width // 4, fb_height // 4
            pip_viewport = mujoco.MjrRect(fb_width - pip_w - 10, 10, pip_w, pip_h)

            # Update robot camera — true first-person from robot's eyes
            # Get torso world position and robot heading
            torso_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            torso_pos = self.env.data.xpos[torso_id]
            w, x, y, z = (self.env.data.qpos[3], self.env.data.qpos[4],
                           self.env.data.qpos[5], self.env.data.qpos[6])
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

            # Eye position: top of torso + 0.6m up (head), 0.3m forward (past geometry)
            eye_x = torso_pos[0] + 0.3 * math.cos(yaw)
            eye_y = torso_pos[1] + 0.3 * math.sin(yaw)
            eye_z = torso_pos[2] + 0.6

            # Lookat: just barely ahead of the eye (tiny distance = camera at eye)
            self._robot_camera.lookat[0] = eye_x + 0.01 * math.cos(yaw)
            self._robot_camera.lookat[1] = eye_y + 0.01 * math.sin(yaw)
            self._robot_camera.lookat[2] = eye_z - 0.05  # Slight downward gaze
            self._robot_camera.distance = 0.01
            self._robot_camera.azimuth = math.degrees(yaw)
            self._robot_camera.elevation = -10

            mujoco.mjv_updateScene(
                self.env.model, self.env.data, self.option, None,
                self._robot_camera, mujoco.mjtCatBit.mjCAT_ALL, self._robot_scene,
            )
            mujoco.mjr_render(pip_viewport, self._robot_scene, self.context)

            # PiP label
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                pip_viewport, "ROBOT CAM", "", self.context,
            )

            # ── On-screen overlays ──

            # Bottom-left: status + action log
            log_text = self._status_text
            if self._current_item and self._current_step:
                log_text += f"\n  > {self._current_step}: {self._current_item}"
            if self._log_lines:
                log_text += "\n\n" + "\n".join(self._log_lines)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                self.viewport, log_text, "", self.context,
            )

            # Top-left: shopping list with checkmarks
            item_lines = "SHOPPING LIST\n" + "-" * 20 + "\n"
            for i in self.items:
                if i.name in self._fetched:
                    mark = "[DONE]"
                elif i.name == self._current_item:
                    mark = "[>>>]"
                else:
                    mark = "[    ]"
                item_lines += f" {mark} {i.name}\n"
            cart_count = len(self._fetched)
            item_lines += f"\nCart: {cart_count} item{'s' if cart_count != 1 else ''}"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                self.viewport, item_lines, "", self.context,
            )

            # Top-right: big pick notification (fades after 2 seconds)
            if self._pick_notification and (time.time() - self._pick_notification_time) < 2.0:
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    self.viewport, self._pick_notification, "", self.context,
                )

            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()
        self.env.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PantryPilot — Full Pipeline")
    parser.add_argument("--items", nargs="+", default=None,
                        help="Recipe names to cook (e.g. --items carbonara 'stir fry')")
    parser.add_argument("--no-vision", action="store_true",
                        help="Disable VLM shelf scanning (use lookup only)")
    args = parser.parse_args()

    # ── Always start the API server ──
    print("\nStarting PantryPilot API server...")
    start_api_server()

    # ── Phase 1: Get recipes ──
    if args.items:
        recipes = args.items
    else:
        print("\n" + "=" * 60)
        print("  PantryPilot — What do you want to cook?")
        print("=" * 60)
        print("\nEnter the recipes you want to cook (comma-separated):")
        print("Example: carbonara, stir fry, tacos\n")

        recipe_input = input("> ").strip()
        if not recipe_input:
            print("No recipes entered, defaulting to: carbonara")
            recipe_input = "carbonara"
        recipes = [r.strip() for r in recipe_input.split(",")]

    print(f"\nRecipes: {recipes}")

    # ── Phase 2: Call T1 API for shopping list ──
    print("\nAsking Planner API for shopping list...")
    items = get_shopping_list_from_api(recipes)

    if not items:
        print("API returned no matchable items — cannot proceed.")
        print("Check that NEBIUS_API_KEY is set in .env")
        return

    print(f"\n{'=' * 60}")
    print(f"  Shopping List (from API) — {len(items)} items to fetch:")
    print(f"{'=' * 60}")
    for item in items:
        print(f"  • {item.name} ({item.quantity}) — {item.aisle} aisle")
    print()

    print(f"Starting simulation with {len(items)} items...\n")

    env = GroceryStoreEnv()
    env.load()

    use_vision = not args.no_vision
    if use_vision:
        print("VLM vision enabled — robot will use Qwen2.5-VL to scan shelves\n")
    viewer = GroceryViewer(env, items, use_vision=use_vision)
    viewer.run()


if __name__ == "__main__":
    main()
