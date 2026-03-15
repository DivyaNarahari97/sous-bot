"""MuJoCo grocery store environment with real Unitree G1 robot (29-DOF + hands).

Uses the official G1 MJCF from Google DeepMind's MuJoCo Menagerie via robot_descriptions.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field

import mujoco
import numpy as np
from robot_descriptions import g1_mj_description

logger = logging.getLogger(__name__)

# --- G1 model paths ---
_G1_MJCF_DIR = os.path.dirname(g1_mj_description.MJCF_PATH)
_G1_WITH_HANDS_PATH = os.path.join(_G1_MJCF_DIR, "g1_with_hands.xml")

# --- G1 actuator index map (43 actuators total) ---
# Legs: 0-11, Waist: 12-14, Left arm: 15-21, Left hand: 22-28, Right arm: 29-35, Right hand: 36-42
G1_ACTUATORS = {
    # Left leg
    "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
    "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
    # Right leg
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
    "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
    # Waist
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    # Left arm
    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
    "left_elbow": 18, "left_wrist_roll": 19, "left_wrist_pitch": 20, "left_wrist_yaw": 21,
    # Left hand fingers
    "left_thumb_0": 22, "left_thumb_1": 23, "left_thumb_2": 24,
    "left_middle_0": 25, "left_middle_1": 26, "left_index_0": 27, "left_index_1": 28,
    # Right arm
    "right_shoulder_pitch": 29, "right_shoulder_roll": 30, "right_shoulder_yaw": 31,
    "right_elbow": 32, "right_wrist_roll": 33, "right_wrist_pitch": 34, "right_wrist_yaw": 35,
    # Right hand fingers
    "right_thumb_0": 36, "right_thumb_1": 37, "right_thumb_2": 38,
    "right_index_0": 39, "right_index_1": 40, "right_middle_0": 41, "right_middle_1": 42,
}

# Standing pose — joint position targets that keep the G1 upright and stable
G1_STANDING_POSE = {
    # Legs: straighter stance for more stability
    "left_hip_pitch": -0.15, "left_hip_roll": 0.0, "left_hip_yaw": 0.0,
    "left_knee": 0.3, "left_ankle_pitch": -0.15, "left_ankle_roll": 0.0,
    "right_hip_pitch": -0.15, "right_hip_roll": 0.0, "right_hip_yaw": 0.0,
    "right_knee": 0.3, "right_ankle_pitch": -0.15, "right_ankle_roll": 0.0,
    # Waist upright
    "waist_yaw": 0.0, "waist_roll": 0.0, "waist_pitch": 0.0,
    # Arms relaxed at sides
    "left_shoulder_pitch": 0.3, "left_shoulder_roll": 0.15, "left_elbow": 0.3,
    "right_shoulder_pitch": 0.3, "right_shoulder_roll": -0.15, "right_elbow": 0.3,
}

# Initial height for the G1's pelvis (freejoint qpos[2]) so feet are on the floor
G1_INITIAL_HEIGHT = 0.75

# --- Grocery store layout ---
AISLE_NAMES = ["produce", "dairy", "bakery", "deli", "spices", "frozen"]

# Shelf heights: bottom=0.42, middle=0.77, top=1.12
_SHELF_Z = {0: 0.42, 1: 0.77, 2: 1.12}

# Left shelves at y = aisle_y + 2.0, right shelves at y = aisle_y - 2.0
# Multiple items per shelf, spread along x-axis with spacing

def _shelf_pos(aisle_x: float, side: str, shelf: int, offset: float = 0.0) -> tuple:
    """Compute item position on a shelf. offset spreads items along x."""
    y = 1.8 if side == "left" else -1.8
    return (aisle_x + offset, y, _SHELF_Z[shelf])

STORE_ITEMS: dict[str, dict] = {
    # ═══ PRODUCE AISLE (x=1.0) — fresh fruits, vegetables, oils ═══
    # Left shelf bottom
    "tomatoes":       {"aisle": "produce", "shelf": 0, "side": "left",  "position": _shelf_pos(1.0, "left", 0, -0.3)},
    "onion":          {"aisle": "produce", "shelf": 0, "side": "left",  "position": _shelf_pos(1.0, "left", 0, -0.1)},
    "garlic":         {"aisle": "produce", "shelf": 0, "side": "left",  "position": _shelf_pos(1.0, "left", 0, 0.1)},
    "potato":         {"aisle": "produce", "shelf": 0, "side": "left",  "position": _shelf_pos(1.0, "left", 0, 0.3)},
    # Left shelf middle
    "lemon":          {"aisle": "produce", "shelf": 1, "side": "left",  "position": _shelf_pos(1.0, "left", 1, -0.3)},
    "lime":           {"aisle": "produce", "shelf": 1, "side": "left",  "position": _shelf_pos(1.0, "left", 1, -0.1)},
    "olive oil":      {"aisle": "produce", "shelf": 1, "side": "left",  "position": _shelf_pos(1.0, "left", 1, 0.1)},
    "avocado":        {"aisle": "produce", "shelf": 1, "side": "left",  "position": _shelf_pos(1.0, "left", 1, 0.3)},
    # Left shelf top
    "banana":         {"aisle": "produce", "shelf": 2, "side": "left",  "position": _shelf_pos(1.0, "left", 2, -0.2)},
    "apple":          {"aisle": "produce", "shelf": 2, "side": "left",  "position": _shelf_pos(1.0, "left", 2, 0.0)},
    "orange":         {"aisle": "produce", "shelf": 2, "side": "left",  "position": _shelf_pos(1.0, "left", 2, 0.2)},
    # Right shelf bottom
    "bell pepper":    {"aisle": "produce", "shelf": 0, "side": "right", "position": _shelf_pos(1.0, "right", 0, -0.3)},
    "lettuce":        {"aisle": "produce", "shelf": 0, "side": "right", "position": _shelf_pos(1.0, "right", 0, -0.1)},
    "cucumber":       {"aisle": "produce", "shelf": 0, "side": "right", "position": _shelf_pos(1.0, "right", 0, 0.1)},
    "carrot":         {"aisle": "produce", "shelf": 0, "side": "right", "position": _shelf_pos(1.0, "right", 0, 0.3)},
    # Right shelf middle
    "ginger":         {"aisle": "produce", "shelf": 1, "side": "right", "position": _shelf_pos(1.0, "right", 1, -0.2)},
    "cilantro":       {"aisle": "produce", "shelf": 1, "side": "right", "position": _shelf_pos(1.0, "right", 1, 0.0)},
    "spinach":        {"aisle": "produce", "shelf": 1, "side": "right", "position": _shelf_pos(1.0, "right", 1, 0.2)},
    # Right shelf top
    "mushrooms":      {"aisle": "produce", "shelf": 2, "side": "right", "position": _shelf_pos(1.0, "right", 2, -0.15)},
    "green onion":    {"aisle": "produce", "shelf": 2, "side": "right", "position": _shelf_pos(1.0, "right", 2, 0.15)},

    # ═══ DAIRY AISLE (x=2.0) — milk, eggs, cheese, butter ═══
    # Left shelf bottom
    "milk":           {"aisle": "dairy", "shelf": 0, "side": "left",  "position": _shelf_pos(2.0, "left", 0, -0.3)},
    "almond milk":    {"aisle": "dairy", "shelf": 0, "side": "left",  "position": _shelf_pos(2.0, "left", 0, -0.1)},
    "eggs":           {"aisle": "dairy", "shelf": 0, "side": "left",  "position": _shelf_pos(2.0, "left", 0, 0.1)},
    "egg whites":     {"aisle": "dairy", "shelf": 0, "side": "left",  "position": _shelf_pos(2.0, "left", 0, 0.3)},
    # Left shelf middle
    "heavy cream":    {"aisle": "dairy", "shelf": 1, "side": "left",  "position": _shelf_pos(2.0, "left", 1, -0.2)},
    "sour cream":     {"aisle": "dairy", "shelf": 1, "side": "left",  "position": _shelf_pos(2.0, "left", 1, 0.0)},
    "cream cheese":   {"aisle": "dairy", "shelf": 1, "side": "left",  "position": _shelf_pos(2.0, "left", 1, 0.2)},
    # Left shelf top
    "whipped cream":  {"aisle": "dairy", "shelf": 2, "side": "left",  "position": _shelf_pos(2.0, "left", 2, -0.1)},
    "half and half":  {"aisle": "dairy", "shelf": 2, "side": "left",  "position": _shelf_pos(2.0, "left", 2, 0.1)},
    # Right shelf bottom
    "butter":         {"aisle": "dairy", "shelf": 0, "side": "right", "position": _shelf_pos(2.0, "right", 0, -0.3)},
    "margarine":      {"aisle": "dairy", "shelf": 0, "side": "right", "position": _shelf_pos(2.0, "right", 0, -0.1)},
    "parmesan":       {"aisle": "dairy", "shelf": 0, "side": "right", "position": _shelf_pos(2.0, "right", 0, 0.1)},
    "pecorino":       {"aisle": "dairy", "shelf": 0, "side": "right", "position": _shelf_pos(2.0, "right", 0, 0.3)},
    # Right shelf middle
    "mozzarella":     {"aisle": "dairy", "shelf": 1, "side": "right", "position": _shelf_pos(2.0, "right", 1, -0.2)},
    "cheddar":        {"aisle": "dairy", "shelf": 1, "side": "right", "position": _shelf_pos(2.0, "right", 1, 0.0)},
    "swiss cheese":   {"aisle": "dairy", "shelf": 1, "side": "right", "position": _shelf_pos(2.0, "right", 1, 0.2)},
    # Right shelf top
    "yogurt":         {"aisle": "dairy", "shelf": 2, "side": "right", "position": _shelf_pos(2.0, "right", 2, -0.15)},
    "greek yogurt":   {"aisle": "dairy", "shelf": 2, "side": "right", "position": _shelf_pos(2.0, "right", 2, 0.15)},

    # ═══ BAKERY AISLE (x=3.0) — bread, pasta, grains ═══
    # Left shelf bottom
    "bread":          {"aisle": "bakery", "shelf": 0, "side": "left",  "position": _shelf_pos(3.0, "left", 0, -0.25)},
    "sourdough":      {"aisle": "bakery", "shelf": 0, "side": "left",  "position": _shelf_pos(3.0, "left", 0, 0.0)},
    "tortillas":      {"aisle": "bakery", "shelf": 0, "side": "left",  "position": _shelf_pos(3.0, "left", 0, 0.25)},
    # Left shelf middle
    "pasta":          {"aisle": "bakery", "shelf": 1, "side": "left",  "position": _shelf_pos(3.0, "left", 1, -0.25)},
    "penne":          {"aisle": "bakery", "shelf": 1, "side": "left",  "position": _shelf_pos(3.0, "left", 1, 0.0)},
    "rice":           {"aisle": "bakery", "shelf": 1, "side": "left",  "position": _shelf_pos(3.0, "left", 1, 0.25)},
    # Left shelf top
    "noodles":        {"aisle": "bakery", "shelf": 2, "side": "left",  "position": _shelf_pos(3.0, "left", 2, -0.15)},
    "couscous":       {"aisle": "bakery", "shelf": 2, "side": "left",  "position": _shelf_pos(3.0, "left", 2, 0.15)},
    # Right shelf bottom
    "flour":          {"aisle": "bakery", "shelf": 0, "side": "right", "position": _shelf_pos(3.0, "right", 0, -0.2)},
    "cornstarch":     {"aisle": "bakery", "shelf": 0, "side": "right", "position": _shelf_pos(3.0, "right", 0, 0.0)},
    "baking powder":  {"aisle": "bakery", "shelf": 0, "side": "right", "position": _shelf_pos(3.0, "right", 0, 0.2)},
    # Right shelf middle
    "sugar":          {"aisle": "bakery", "shelf": 1, "side": "right", "position": _shelf_pos(3.0, "right", 1, -0.2)},
    "brown sugar":    {"aisle": "bakery", "shelf": 1, "side": "right", "position": _shelf_pos(3.0, "right", 1, 0.0)},
    "honey":          {"aisle": "bakery", "shelf": 1, "side": "right", "position": _shelf_pos(3.0, "right", 1, 0.2)},
    # Right shelf top
    "maple syrup":    {"aisle": "bakery", "shelf": 2, "side": "right", "position": _shelf_pos(3.0, "right", 2, 0.0)},

    # ═══ DELI / MEAT AISLE (x=4.0) ═══
    # Left shelf bottom
    "guanciale":      {"aisle": "deli", "shelf": 0, "side": "left",  "position": _shelf_pos(4.0, "left", 0, -0.3)},
    "pancetta":       {"aisle": "deli", "shelf": 0, "side": "left",  "position": _shelf_pos(4.0, "left", 0, -0.1)},
    "prosciutto":     {"aisle": "deli", "shelf": 0, "side": "left",  "position": _shelf_pos(4.0, "left", 0, 0.1)},
    "salami":         {"aisle": "deli", "shelf": 0, "side": "left",  "position": _shelf_pos(4.0, "left", 0, 0.3)},
    # Left shelf middle
    "chicken breast": {"aisle": "deli", "shelf": 1, "side": "left",  "position": _shelf_pos(4.0, "left", 1, -0.25)},
    "chicken thigh":  {"aisle": "deli", "shelf": 1, "side": "left",  "position": _shelf_pos(4.0, "left", 1, 0.0)},
    "ground beef":    {"aisle": "deli", "shelf": 1, "side": "left",  "position": _shelf_pos(4.0, "left", 1, 0.25)},
    # Left shelf top
    "ground turkey":  {"aisle": "deli", "shelf": 2, "side": "left",  "position": _shelf_pos(4.0, "left", 2, -0.1)},
    "sausage":        {"aisle": "deli", "shelf": 2, "side": "left",  "position": _shelf_pos(4.0, "left", 2, 0.1)},
    # Right shelf bottom
    "bacon":          {"aisle": "deli", "shelf": 0, "side": "right", "position": _shelf_pos(4.0, "right", 0, -0.25)},
    "ham":            {"aisle": "deli", "shelf": 0, "side": "right", "position": _shelf_pos(4.0, "right", 0, 0.0)},
    "salmon":         {"aisle": "deli", "shelf": 0, "side": "right", "position": _shelf_pos(4.0, "right", 0, 0.25)},
    # Right shelf middle
    "shrimp":         {"aisle": "deli", "shelf": 1, "side": "right", "position": _shelf_pos(4.0, "right", 1, -0.2)},
    "tofu":           {"aisle": "deli", "shelf": 1, "side": "right", "position": _shelf_pos(4.0, "right", 1, 0.0)},
    "tempeh":         {"aisle": "deli", "shelf": 1, "side": "right", "position": _shelf_pos(4.0, "right", 1, 0.2)},
    # Right shelf top
    "tuna can":       {"aisle": "deli", "shelf": 2, "side": "right", "position": _shelf_pos(4.0, "right", 2, -0.1)},
    "sardines":       {"aisle": "deli", "shelf": 2, "side": "right", "position": _shelf_pos(4.0, "right", 2, 0.1)},

    # ═══ SPICES / CONDIMENTS AISLE (x=5.0) ═══
    # Left shelf bottom
    "black pepper":   {"aisle": "spices", "shelf": 0, "side": "left",  "position": _shelf_pos(5.0, "left", 0, -0.3)},
    "salt":           {"aisle": "spices", "shelf": 0, "side": "left",  "position": _shelf_pos(5.0, "left", 0, -0.15)},
    "cumin":          {"aisle": "spices", "shelf": 0, "side": "left",  "position": _shelf_pos(5.0, "left", 0, 0.0)},
    "paprika":        {"aisle": "spices", "shelf": 0, "side": "left",  "position": _shelf_pos(5.0, "left", 0, 0.15)},
    "chili flakes":   {"aisle": "spices", "shelf": 0, "side": "left",  "position": _shelf_pos(5.0, "left", 0, 0.3)},
    # Left shelf middle
    "oregano":        {"aisle": "spices", "shelf": 1, "side": "left",  "position": _shelf_pos(5.0, "left", 1, -0.3)},
    "basil":          {"aisle": "spices", "shelf": 1, "side": "left",  "position": _shelf_pos(5.0, "left", 1, -0.15)},
    "thyme":          {"aisle": "spices", "shelf": 1, "side": "left",  "position": _shelf_pos(5.0, "left", 1, 0.0)},
    "cinnamon":       {"aisle": "spices", "shelf": 1, "side": "left",  "position": _shelf_pos(5.0, "left", 1, 0.15)},
    "turmeric":       {"aisle": "spices", "shelf": 1, "side": "left",  "position": _shelf_pos(5.0, "left", 1, 0.3)},
    # Left shelf top
    "soy sauce":      {"aisle": "spices", "shelf": 2, "side": "left",  "position": _shelf_pos(5.0, "left", 2, -0.2)},
    "fish sauce":     {"aisle": "spices", "shelf": 2, "side": "left",  "position": _shelf_pos(5.0, "left", 2, 0.0)},
    "hot sauce":      {"aisle": "spices", "shelf": 2, "side": "left",  "position": _shelf_pos(5.0, "left", 2, 0.2)},
    # Right shelf bottom
    "salsa":          {"aisle": "spices", "shelf": 0, "side": "right", "position": _shelf_pos(5.0, "right", 0, -0.25)},
    "mustard":        {"aisle": "spices", "shelf": 0, "side": "right", "position": _shelf_pos(5.0, "right", 0, 0.0)},
    "mayo":           {"aisle": "spices", "shelf": 0, "side": "right", "position": _shelf_pos(5.0, "right", 0, 0.25)},
    # Right shelf middle
    "ketchup":        {"aisle": "spices", "shelf": 1, "side": "right", "position": _shelf_pos(5.0, "right", 1, -0.2)},
    "bbq sauce":      {"aisle": "spices", "shelf": 1, "side": "right", "position": _shelf_pos(5.0, "right", 1, 0.0)},
    "vinegar":        {"aisle": "spices", "shelf": 1, "side": "right", "position": _shelf_pos(5.0, "right", 1, 0.2)},
    # Right shelf top
    "sesame oil":     {"aisle": "spices", "shelf": 2, "side": "right", "position": _shelf_pos(5.0, "right", 2, -0.1)},
    "coconut oil":    {"aisle": "spices", "shelf": 2, "side": "right", "position": _shelf_pos(5.0, "right", 2, 0.1)},

    # ═══ FROZEN / SNACKS AISLE (x=6.0) ═══
    # Left shelf bottom
    "frozen peas":    {"aisle": "frozen", "shelf": 0, "side": "left",  "position": _shelf_pos(6.0, "left", 0, -0.25)},
    "frozen corn":    {"aisle": "frozen", "shelf": 0, "side": "left",  "position": _shelf_pos(6.0, "left", 0, 0.0)},
    "ice cream":      {"aisle": "frozen", "shelf": 0, "side": "left",  "position": _shelf_pos(6.0, "left", 0, 0.25)},
    # Left shelf middle
    "frozen pizza":   {"aisle": "frozen", "shelf": 1, "side": "left",  "position": _shelf_pos(6.0, "left", 1, -0.2)},
    "frozen fries":   {"aisle": "frozen", "shelf": 1, "side": "left",  "position": _shelf_pos(6.0, "left", 1, 0.0)},
    "frozen berries": {"aisle": "frozen", "shelf": 1, "side": "left",  "position": _shelf_pos(6.0, "left", 1, 0.2)},
    # Left shelf top
    "frozen waffles": {"aisle": "frozen", "shelf": 2, "side": "left",  "position": _shelf_pos(6.0, "left", 2, -0.1)},
    "frozen burritos":{"aisle": "frozen", "shelf": 2, "side": "left",  "position": _shelf_pos(6.0, "left", 2, 0.1)},
    # Right shelf bottom
    "chips":          {"aisle": "frozen", "shelf": 0, "side": "right", "position": _shelf_pos(6.0, "right", 0, -0.25)},
    "pretzels":       {"aisle": "frozen", "shelf": 0, "side": "right", "position": _shelf_pos(6.0, "right", 0, 0.0)},
    "crackers":       {"aisle": "frozen", "shelf": 0, "side": "right", "position": _shelf_pos(6.0, "right", 0, 0.25)},
    # Right shelf middle
    "cookies":        {"aisle": "frozen", "shelf": 1, "side": "right", "position": _shelf_pos(6.0, "right", 1, -0.2)},
    "granola bars":   {"aisle": "frozen", "shelf": 1, "side": "right", "position": _shelf_pos(6.0, "right", 1, 0.0)},
    "popcorn":        {"aisle": "frozen", "shelf": 1, "side": "right", "position": _shelf_pos(6.0, "right", 1, 0.2)},
    # Right shelf top
    "trail mix":      {"aisle": "frozen", "shelf": 2, "side": "right", "position": _shelf_pos(6.0, "right", 2, -0.1)},
    "nuts":           {"aisle": "frozen", "shelf": 2, "side": "right", "position": _shelf_pos(6.0, "right", 2, 0.1)},
}

AISLE_POSITIONS: dict[str, tuple[float, float]] = {
    "produce": (1.0, 0.0),
    "dairy": (2.0, 0.0),
    "bakery": (3.0, 0.0),
    "deli": (4.0, 0.0),
    "spices": (5.0, 0.0),
    "frozen": (6.0, 0.0),
}

CART_POSITION = (0.0, 0.0, 0.0)
CHECKOUT_POSITION = (7.0, 0.0)
ENTRANCE_POSITION = (0.0, 0.0)


def _build_grocery_store_xml() -> str:
    """Generate MuJoCo XML for grocery store + real Unitree G1 robot."""
    shelves_xml = ""
    # Shelf dimensions: back panel + 3 horizontal shelf boards
    # Back panel: tall and thin. Shelves: wide platforms extending toward the aisle.
    for aisle_name, (ax, ay) in AISLE_POSITIONS.items():
        # Left shelf (at y = aisle_y + 2.0, items face toward y-negative / aisle center)
        shelves_xml += f"""
        <body name="shelf_{aisle_name}_left" pos="{ax} {ay + 2.0} 0">
            <!-- Back panel -->
            <geom type="box" size="0.5 0.05 0.7" pos="0 0.05 0.7" rgba="0.55 0.35 0.18 1" />
            <!-- Bottom shelf -->
            <geom type="box" size="0.5 0.2 0.015" pos="0 -0.1 {_SHELF_Z[0] - 0.06}" rgba="0.65 0.45 0.25 1" />
            <!-- Middle shelf -->
            <geom type="box" size="0.5 0.2 0.015" pos="0 -0.1 {_SHELF_Z[1] - 0.06}" rgba="0.65 0.45 0.25 1" />
            <!-- Top shelf -->
            <geom type="box" size="0.5 0.2 0.015" pos="0 -0.1 {_SHELF_Z[2] - 0.06}" rgba="0.65 0.45 0.25 1" />
        </body>
        <!-- Right shelf (at y = aisle_y - 2.0, items face toward y-positive / aisle center) -->
        <body name="shelf_{aisle_name}_right" pos="{ax} {ay - 2.0} 0">
            <!-- Back panel -->
            <geom type="box" size="0.5 0.05 0.7" pos="0 -0.05 0.7" rgba="0.55 0.35 0.18 1" />
            <!-- Bottom shelf -->
            <geom type="box" size="0.5 0.2 0.015" pos="0 0.1 {_SHELF_Z[0] - 0.06}" rgba="0.65 0.45 0.25 1" />
            <!-- Middle shelf -->
            <geom type="box" size="0.5 0.2 0.015" pos="0 0.1 {_SHELF_Z[1] - 0.06}" rgba="0.65 0.45 0.25 1" />
            <!-- Top shelf -->
            <geom type="box" size="0.5 0.2 0.015" pos="0 0.1 {_SHELF_Z[2] - 0.06}" rgba="0.65 0.45 0.25 1" />
        </body>"""

    items_xml = ""
    # Per-item shapes and colors — realistic grocery visuals
    # Default fallback for items without explicit visuals
    _default_vis = {"type": "box", "size": "0.035 0.025 0.04", "rgba": "0.6 0.6 0.6 1"}
    item_visuals: dict[str, dict] = {
        # ── Produce ──
        "tomatoes":      {"type": "sphere",   "size": "0.035",          "rgba": "0.9 0.12 0.1 1"},
        "onion":         {"type": "sphere",   "size": "0.035",          "rgba": "0.85 0.75 0.3 1"},
        "garlic":        {"type": "sphere",   "size": "0.02",           "rgba": "0.95 0.93 0.85 1"},
        "potato":        {"type": "ellipsoid","size": "0.035 0.025 0.025","rgba": "0.7 0.55 0.3 1"},
        "lemon":         {"type": "sphere",   "size": "0.025",          "rgba": "1.0 0.92 0.2 1"},
        "lime":          {"type": "sphere",   "size": "0.025",          "rgba": "0.3 0.8 0.15 1"},
        "olive oil":     {"type": "cylinder", "size": "0.02 0.06",      "rgba": "0.65 0.7 0.15 1"},
        "avocado":       {"type": "ellipsoid","size": "0.025 0.02 0.035","rgba": "0.2 0.35 0.15 1"},
        "banana":        {"type": "ellipsoid","size": "0.015 0.015 0.05","rgba": "1.0 0.9 0.2 1"},
        "apple":         {"type": "sphere",   "size": "0.03",           "rgba": "0.85 0.15 0.1 1"},
        "orange":        {"type": "sphere",   "size": "0.03",           "rgba": "1.0 0.55 0.0 1"},
        "bell pepper":   {"type": "sphere",   "size": "0.035",          "rgba": "0.1 0.7 0.15 1"},
        "lettuce":       {"type": "sphere",   "size": "0.04",           "rgba": "0.3 0.75 0.2 1"},
        "cucumber":      {"type": "cylinder", "size": "0.02 0.05",      "rgba": "0.2 0.6 0.15 1"},
        "carrot":        {"type": "cylinder", "size": "0.012 0.05",     "rgba": "1.0 0.5 0.05 1"},
        "ginger":        {"type": "ellipsoid","size": "0.03 0.015 0.015","rgba": "0.8 0.65 0.35 1"},
        "cilantro":      {"type": "cylinder", "size": "0.015 0.035",    "rgba": "0.25 0.65 0.2 1"},
        "spinach":       {"type": "box",      "size": "0.035 0.025 0.03","rgba": "0.15 0.5 0.12 1"},
        "mushrooms":     {"type": "sphere",   "size": "0.025",          "rgba": "0.85 0.78 0.65 1"},
        "green onion":   {"type": "cylinder", "size": "0.008 0.05",     "rgba": "0.2 0.6 0.15 1"},
        # ── Dairy ──
        "milk":          {"type": "cylinder", "size": "0.025 0.065",    "rgba": "0.95 0.95 0.97 1"},
        "almond milk":   {"type": "cylinder", "size": "0.025 0.065",    "rgba": "0.9 0.85 0.7 1"},
        "eggs":          {"type": "box",      "size": "0.05 0.035 0.02","rgba": "0.95 0.92 0.85 1"},
        "egg whites":    {"type": "cylinder", "size": "0.025 0.05",     "rgba": "0.97 0.97 0.9 1"},
        "heavy cream":   {"type": "cylinder", "size": "0.02 0.045",     "rgba": "0.98 0.96 0.9 1"},
        "sour cream":    {"type": "cylinder", "size": "0.025 0.025",    "rgba": "0.97 0.97 0.95 1"},
        "cream cheese":  {"type": "box",      "size": "0.04 0.02 0.025","rgba": "0.97 0.95 0.9 1"},
        "whipped cream": {"type": "cylinder", "size": "0.02 0.055",     "rgba": "0.98 0.98 0.98 1"},
        "half and half": {"type": "cylinder", "size": "0.02 0.045",     "rgba": "0.95 0.93 0.88 1"},
        "butter":        {"type": "box",      "size": "0.035 0.02 0.015","rgba": "1.0 0.92 0.35 1"},
        "margarine":     {"type": "box",      "size": "0.035 0.02 0.015","rgba": "0.95 0.88 0.3 1"},
        "parmesan":      {"type": "cylinder", "size": "0.03 0.025",     "rgba": "0.95 0.88 0.55 1"},
        "pecorino":      {"type": "cylinder", "size": "0.03 0.025",     "rgba": "0.92 0.85 0.5 1"},
        "mozzarella":    {"type": "sphere",   "size": "0.03",           "rgba": "0.97 0.97 0.92 1"},
        "cheddar":       {"type": "box",      "size": "0.035 0.025 0.025","rgba": "1.0 0.7 0.1 1"},
        "swiss cheese":  {"type": "box",      "size": "0.035 0.025 0.025","rgba": "0.95 0.9 0.6 1"},
        "yogurt":        {"type": "cylinder", "size": "0.025 0.03",     "rgba": "0.95 0.92 0.95 1"},
        "greek yogurt":  {"type": "cylinder", "size": "0.025 0.03",     "rgba": "0.9 0.88 0.95 1"},
        # ── Bakery ──
        "bread":         {"type": "ellipsoid","size": "0.06 0.03 0.03", "rgba": "0.85 0.65 0.3 1"},
        "sourdough":     {"type": "ellipsoid","size": "0.06 0.03 0.035","rgba": "0.8 0.6 0.25 1"},
        "tortillas":     {"type": "cylinder", "size": "0.045 0.012",    "rgba": "0.95 0.9 0.75 1"},
        "pasta":         {"type": "box",      "size": "0.03 0.015 0.06","rgba": "0.9 0.82 0.45 1"},
        "penne":         {"type": "box",      "size": "0.03 0.015 0.06","rgba": "0.88 0.8 0.4 1"},
        "rice":          {"type": "box",      "size": "0.035 0.02 0.05","rgba": "0.95 0.95 0.9 1"},
        "noodles":       {"type": "box",      "size": "0.03 0.02 0.055","rgba": "0.92 0.85 0.5 1"},
        "couscous":      {"type": "box",      "size": "0.03 0.02 0.04", "rgba": "0.93 0.88 0.6 1"},
        "flour":         {"type": "box",      "size": "0.035 0.025 0.05","rgba": "0.97 0.97 0.95 1"},
        "cornstarch":    {"type": "box",      "size": "0.03 0.02 0.045","rgba": "0.95 0.95 0.92 1"},
        "baking powder": {"type": "cylinder", "size": "0.02 0.035",     "rgba": "0.9 0.45 0.1 1"},
        "sugar":         {"type": "box",      "size": "0.03 0.02 0.04", "rgba": "0.98 0.98 0.98 1"},
        "brown sugar":   {"type": "box",      "size": "0.03 0.02 0.04", "rgba": "0.65 0.4 0.15 1"},
        "honey":         {"type": "cylinder", "size": "0.02 0.04",      "rgba": "0.9 0.7 0.1 1"},
        "maple syrup":   {"type": "cylinder", "size": "0.02 0.05",      "rgba": "0.6 0.35 0.1 1"},
        # ── Deli / Meat ──
        "guanciale":     {"type": "box",      "size": "0.04 0.03 0.02", "rgba": "0.85 0.35 0.3 1"},
        "pancetta":      {"type": "cylinder", "size": "0.03 0.02",      "rgba": "0.8 0.3 0.25 1"},
        "prosciutto":    {"type": "box",      "size": "0.045 0.03 0.012","rgba": "0.9 0.45 0.35 1"},
        "salami":        {"type": "cylinder", "size": "0.025 0.04",     "rgba": "0.7 0.2 0.15 1"},
        "chicken breast":{"type": "box",      "size": "0.04 0.035 0.015","rgba": "0.95 0.8 0.7 1"},
        "chicken thigh": {"type": "box",      "size": "0.04 0.035 0.02","rgba": "0.92 0.75 0.6 1"},
        "ground beef":   {"type": "box",      "size": "0.04 0.035 0.025","rgba": "0.6 0.15 0.12 1"},
        "ground turkey": {"type": "box",      "size": "0.04 0.035 0.025","rgba": "0.85 0.65 0.55 1"},
        "sausage":       {"type": "cylinder", "size": "0.015 0.05",     "rgba": "0.7 0.25 0.15 1"},
        "bacon":         {"type": "box",      "size": "0.05 0.025 0.012","rgba": "0.75 0.25 0.2 1"},
        "ham":           {"type": "box",      "size": "0.045 0.03 0.015","rgba": "0.9 0.55 0.5 1"},
        "salmon":        {"type": "box",      "size": "0.05 0.03 0.015","rgba": "0.95 0.55 0.4 1"},
        "shrimp":        {"type": "box",      "size": "0.04 0.03 0.02", "rgba": "0.95 0.7 0.6 1"},
        "tofu":          {"type": "box",      "size": "0.035 0.025 0.025","rgba": "0.95 0.92 0.8 1"},
        "tempeh":        {"type": "box",      "size": "0.04 0.025 0.02","rgba": "0.85 0.75 0.55 1"},
        "tuna can":      {"type": "cylinder", "size": "0.03 0.015",     "rgba": "0.3 0.4 0.7 1"},
        "sardines":      {"type": "box",      "size": "0.04 0.02 0.012","rgba": "0.2 0.35 0.6 1"},
        # ── Spices / Condiments ──
        "black pepper":  {"type": "cylinder", "size": "0.012 0.04",     "rgba": "0.15 0.12 0.1 1"},
        "salt":          {"type": "cylinder", "size": "0.012 0.04",     "rgba": "0.95 0.95 0.95 1"},
        "cumin":         {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.7 0.5 0.2 1"},
        "paprika":       {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.8 0.2 0.1 1"},
        "chili flakes":  {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.85 0.15 0.05 1"},
        "oregano":       {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.35 0.5 0.2 1"},
        "basil":         {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.2 0.55 0.15 1"},
        "thyme":         {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.45 0.5 0.3 1"},
        "cinnamon":      {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.6 0.3 0.1 1"},
        "turmeric":      {"type": "cylinder", "size": "0.012 0.035",    "rgba": "0.9 0.7 0.0 1"},
        "soy sauce":     {"type": "cylinder", "size": "0.015 0.05",     "rgba": "0.2 0.12 0.08 1"},
        "fish sauce":    {"type": "cylinder", "size": "0.015 0.05",     "rgba": "0.45 0.3 0.15 1"},
        "hot sauce":     {"type": "cylinder", "size": "0.012 0.05",     "rgba": "0.9 0.15 0.05 1"},
        "salsa":         {"type": "cylinder", "size": "0.025 0.035",    "rgba": "0.85 0.2 0.1 1"},
        "mustard":       {"type": "cylinder", "size": "0.02 0.04",      "rgba": "0.9 0.8 0.1 1"},
        "mayo":          {"type": "cylinder", "size": "0.025 0.04",     "rgba": "0.97 0.95 0.85 1"},
        "ketchup":       {"type": "cylinder", "size": "0.02 0.05",      "rgba": "0.85 0.1 0.05 1"},
        "bbq sauce":     {"type": "cylinder", "size": "0.02 0.05",      "rgba": "0.4 0.15 0.05 1"},
        "vinegar":       {"type": "cylinder", "size": "0.02 0.055",     "rgba": "0.7 0.55 0.25 1"},
        "sesame oil":    {"type": "cylinder", "size": "0.018 0.045",    "rgba": "0.6 0.45 0.1 1"},
        "coconut oil":   {"type": "cylinder", "size": "0.025 0.035",    "rgba": "0.97 0.97 0.95 1"},
        # ── Frozen / Snacks ──
        "frozen peas":   {"type": "box",      "size": "0.035 0.02 0.04","rgba": "0.2 0.7 0.3 1"},
        "frozen corn":   {"type": "box",      "size": "0.035 0.02 0.04","rgba": "0.95 0.85 0.2 1"},
        "ice cream":     {"type": "cylinder", "size": "0.035 0.035",    "rgba": "0.9 0.75 0.6 1"},
        "frozen pizza":  {"type": "box",      "size": "0.06 0.06 0.012","rgba": "0.85 0.5 0.2 1"},
        "frozen fries":  {"type": "box",      "size": "0.035 0.02 0.05","rgba": "0.9 0.8 0.3 1"},
        "frozen berries":{"type": "box",      "size": "0.035 0.02 0.04","rgba": "0.5 0.1 0.35 1"},
        "frozen waffles":{"type": "box",      "size": "0.04 0.03 0.03","rgba": "0.9 0.8 0.5 1"},
        "frozen burritos":{"type": "cylinder","size": "0.02 0.05",     "rgba": "0.85 0.6 0.3 1"},
        "chips":         {"type": "box",      "size": "0.035 0.015 0.06","rgba": "0.2 0.45 0.9 1"},
        "pretzels":      {"type": "box",      "size": "0.035 0.015 0.055","rgba": "0.75 0.55 0.2 1"},
        "crackers":      {"type": "box",      "size": "0.035 0.015 0.05","rgba": "0.9 0.8 0.5 1"},
        "cookies":       {"type": "box",      "size": "0.035 0.025 0.04","rgba": "0.6 0.35 0.15 1"},
        "granola bars":  {"type": "box",      "size": "0.04 0.015 0.03","rgba": "0.7 0.55 0.25 1"},
        "popcorn":       {"type": "box",      "size": "0.04 0.025 0.05","rgba": "0.95 0.9 0.3 1"},
        "trail mix":     {"type": "box",      "size": "0.035 0.02 0.04","rgba": "0.6 0.45 0.2 1"},
        "nuts":          {"type": "box",      "size": "0.035 0.02 0.035","rgba": "0.7 0.5 0.2 1"},
    }
    stock_id = 0
    for item_name, info in STORE_ITEMS.items():
        ix, iy, iz = info["position"]
        safe_name = item_name.replace(" ", "_")
        vis = item_visuals.get(item_name, _default_vis)
        side = info.get("side", "left")
        # The pickable item (front of shelf, facing the aisle)
        items_xml += f"""
        <body name="item_{safe_name}" pos="{ix} {iy} {iz}">
            <geom type="{vis['type']}" size="{vis['size']}" rgba="{vis['rgba']}" contype="0" conaffinity="0" />
        </body>"""
        # Stock duplicates behind the pickable item (deeper into the shelf)
        # This makes shelves look full — 2 copies stacked behind
        dy_back = 0.08 if side == "left" else -0.08
        for depth in range(1, 3):
            stock_id += 1
            sy = iy + dy_back * depth
            items_xml += f"""
        <body name="stock_{stock_id}" pos="{ix} {sy} {iz}">
            <geom type="{vis['type']}" size="{vis['size']}" rgba="{vis['rgba']}" />
        </body>"""

    return f"""<?xml version="1.0" ?>
<mujoco model="grocery_store_g1">
    <!-- Include the real Unitree G1 robot with dexterous hands -->
    <include file="{_G1_WITH_HANDS_PATH}"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.5 0.5 0.5"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global offwidth="1920" offheight="1080"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
                 width="512" height="3072"/>
        <texture type="2d" name="grid" builtin="checker"
                 rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8"
                 width="300" height="300" />
        <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.2" />
    </asset>

    <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" size="10 10 0.05" material="floor_mat" />

        <!-- Lighting -->
        <light directional="true" pos="3 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8" />
        <light directional="true" pos="3 3 4" dir="0 -0.5 -1" diffuse="0.4 0.4 0.4" />

        <!-- Shopping cart (wireframe style) -->
        <body name="cart" pos="0.5 -0.3 0.0">
            <!-- Basket -->
            <geom type="box" size="0.2 0.15 0.01" pos="0 0 0.35" rgba="0.7 0.7 0.7 1" />
            <!-- Basket sides -->
            <geom type="box" size="0.2 0.01 0.1" pos="0 0.14 0.45" rgba="0.6 0.6 0.6 1" />
            <geom type="box" size="0.2 0.01 0.1" pos="0 -0.14 0.45" rgba="0.6 0.6 0.6 1" />
            <geom type="box" size="0.01 0.15 0.1" pos="0.19 0 0.45" rgba="0.6 0.6 0.6 1" />
            <geom type="box" size="0.01 0.15 0.1" pos="-0.19 0 0.45" rgba="0.6 0.6 0.6 1" />
            <!-- Legs -->
            <geom type="cylinder" size="0.015 0.17" pos="0.15 0.1 0.17" rgba="0.5 0.5 0.5 1" />
            <geom type="cylinder" size="0.015 0.17" pos="-0.15 0.1 0.17" rgba="0.5 0.5 0.5 1" />
            <geom type="cylinder" size="0.015 0.17" pos="0.15 -0.1 0.17" rgba="0.5 0.5 0.5 1" />
            <geom type="cylinder" size="0.015 0.17" pos="-0.15 -0.1 0.17" rgba="0.5 0.5 0.5 1" />
            <!-- Wheels -->
            <geom type="sphere" size="0.025" pos="0.15 0.1 0.0" rgba="0.2 0.2 0.2 1" />
            <geom type="sphere" size="0.025" pos="-0.15 0.1 0.0" rgba="0.2 0.2 0.2 1" />
            <geom type="sphere" size="0.025" pos="0.15 -0.1 0.0" rgba="0.2 0.2 0.2 1" />
            <geom type="sphere" size="0.025" pos="-0.15 -0.1 0.0" rgba="0.2 0.2 0.2 1" />
            <!-- Handle -->
            <geom type="capsule" size="0.012 0.12" pos="-0.19 0 0.65" euler="1.57 0 0" rgba="0.3 0.3 0.3 1" />
        </body>

        <!-- Store shelves -->
        {shelves_xml}

        <!-- Grocery items -->
        {items_xml}

        <!-- Store walls -->
        <body name="wall_back" pos="7.5 0 1.5">
            <geom type="box" size="0.1 4 1.5" rgba="0.85 0.85 0.8 1" />
        </body>
        <body name="wall_left" pos="3.5 4 1.5">
            <geom type="box" size="4 0.1 1.5" rgba="0.85 0.85 0.8 1" />
        </body>
        <body name="wall_right" pos="3.5 -4 1.5">
            <geom type="box" size="4 0.1 1.5" rgba="0.85 0.85 0.8 1" />
        </body>
    </worldbody>

    <sensor>
        <framepos name="robot_pos" objtype="body" objname="pelvis" />
        <framepos name="r_hand_pos" objtype="body" objname="right_wrist_yaw_link" />
        <framepos name="l_hand_pos" objtype="body" objname="left_wrist_yaw_link" />
    </sensor>
</mujoco>"""


@dataclass
class GroceryStoreEnv:
    """MuJoCo grocery store with real Unitree G1 (29-DOF + dexterous hands)."""

    model: mujoco.MjModel | None = None
    data: mujoco.MjData | None = None
    renderer: mujoco.Renderer | None = None
    _xml_path: str | None = None
    _items_on_shelf: dict[str, bool] = field(default_factory=lambda: {k: True for k in STORE_ITEMS})
    _items_in_cart: list[str] = field(default_factory=list)
    _held_item_body_id: int = -1  # Body ID of item currently "in hand"

    def load(self) -> None:
        """Load the MuJoCo model (G1 + grocery store) and initialize."""
        xml = _build_grocery_store_xml()
        # Write to temp file in G1 MJCF dir so mesh paths resolve correctly
        with tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False, dir=_G1_MJCF_DIR,
        ) as f:
            f.write(xml)
            self._xml_path = f.name

        self.model = mujoco.MjModel.from_xml_path(self._xml_path)
        self.data = mujoco.MjData(self.model)

        # Set initial robot height so feet are on the floor
        self.data.qpos[2] = G1_INITIAL_HEIGHT

        # Increase control gains for more stable standing
        # Default kp from the model may be too low — boost them
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = max(self.model.actuator_gainprm[i, 0], 100.0)
            self.model.actuator_biasprm[i, 1] = min(self.model.actuator_biasprm[i, 1], -100.0)

        # Set standing pose before first forward pass
        self._apply_standing_pose()
        mujoco.mj_forward(self.model, self.data)

        # Let the robot settle for a few hundred steps
        for _ in range(500):
            self._apply_standing_pose()
            mujoco.mj_step(self.model, self.data)

        logger.info(
            "Grocery store loaded with Unitree G1: %d bodies, %d joints, %d actuators",
            self.model.nbody, self.model.njnt, self.model.nu,
        )

    def _apply_standing_pose(self) -> None:
        """Set actuator targets to standing pose so G1 doesn't collapse."""
        assert self.data is not None
        for name, value in G1_STANDING_POSE.items():
            idx = G1_ACTUATORS.get(name)
            if idx is not None and idx < self.model.nu:
                self.data.ctrl[idx] = value

    def step(self, n_steps: int = 1) -> None:
        """Step the simulation forward."""
        assert self.model is not None and self.data is not None
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
        # Track held item to hand — no mj_forward(), just update body_pos
        # which takes effect on the next mj_step naturally.
        # Items have contype=0 conaffinity=0 so no collision interference.
        if self._held_item_body_id >= 0:
            hand_pos = self.get_hand_position("right")
            self.model.body_pos[self._held_item_body_id] = hand_pos

    def get_robot_position(self) -> np.ndarray:
        """Get current pelvis (base) position [x, y, z]."""
        assert self.data is not None
        return self.data.sensor("robot_pos").data.copy()

    def get_hand_position(self, hand: str = "right") -> np.ndarray:
        """Get current hand palm position [x, y, z]."""
        assert self.data is not None
        sensor_name = "r_hand_pos" if hand == "right" else "l_hand_pos"
        return self.data.sensor(sensor_name).data.copy()

    def set_robot_velocity(self, vx: float, vy: float, vyaw: float = 0.0) -> None:
        """Move the robot by directly setting pelvis qpos (teleport-style for demo).

        The G1 uses position-controlled joints (no velocity actuators on the base).
        For the hackathon demo, we move the floating base directly.
        """
        assert self.model is not None and self.data is not None
        # floating_base_joint is joint 0, a freejoint with 7 qpos values (pos + quat)
        dt = self.model.opt.timestep
        self.data.qpos[0] += vx * dt  # x
        self.data.qpos[1] += vy * dt  # y
        # Keep standing pose active
        self._apply_standing_pose()

    def set_robot_heading(self, yaw: float) -> None:
        """Set robot orientation (yaw) via quaternion on the freejoint.

        qpos[3:7] is the quaternion [w, x, y, z] for the floating base.
        We rotate around the z-axis (vertical).
        """
        assert self.data is not None
        import math
        self.data.qpos[3] = math.cos(yaw / 2)  # w
        self.data.qpos[4] = 0.0                 # x
        self.data.qpos[5] = 0.0                 # y
        self.data.qpos[6] = math.sin(yaw / 2)   # z

    def set_arm_targets(
        self,
        shoulder_pitch: float,
        shoulder_roll: float,
        shoulder_yaw: float,
        elbow: float,
        wrist_roll: float = 0.0,
        wrist_pitch: float = 0.0,
        wrist_yaw: float = 0.0,
        hand: str = "right",
    ) -> None:
        """Set target joint angles for an arm using G1's real actuators."""
        assert self.data is not None
        prefix = "right" if hand == "right" else "left"
        targets = {
            f"{prefix}_shoulder_pitch": shoulder_pitch,
            f"{prefix}_shoulder_roll": shoulder_roll,
            f"{prefix}_shoulder_yaw": shoulder_yaw,
            f"{prefix}_elbow": elbow,
            f"{prefix}_wrist_roll": wrist_roll,
            f"{prefix}_wrist_pitch": wrist_pitch,
            f"{prefix}_wrist_yaw": wrist_yaw,
        }
        for name, value in targets.items():
            idx = G1_ACTUATORS.get(name)
            if idx is not None:
                self.data.ctrl[idx] = value

    def solve_ik_right_arm(self, target_pos: np.ndarray, n_iter: int = 50,
                           step_size: float = 0.3) -> tuple | None:
        """Solve IK for right arm to reach target_pos [x,y,z] using MuJoCo Jacobian.

        Returns a 7-tuple of joint angles (shoulder_pitch, shoulder_roll,
        shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw) or None on failure.
        """
        assert self.model is not None and self.data is not None

        # Right arm actuator indices → joint/qpos indices
        arm_actuator_names = [
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
            "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        ]
        act_ids = [G1_ACTUATORS[n] for n in arm_actuator_names]

        # Map actuator → qpos index via actuator_trnid (joint id → qposadr)
        qpos_ids = []
        for aid in act_ids:
            jnt_id = self.model.actuator_trnid[aid, 0]
            qpos_ids.append(self.model.jnt_qposadr[jnt_id])

        hand_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        if hand_body_id < 0:
            return None

        # Save original qpos
        orig_qpos = self.data.qpos.copy()

        nv = self.model.nv
        jacp = np.zeros((3, nv))

        for _ in range(n_iter):
            mujoco.mj_forward(self.model, self.data)
            hand_pos = self.data.xpos[hand_body_id]
            error = target_pos - hand_pos
            if np.linalg.norm(error) < 0.01:
                break

            # Compute position Jacobian for the hand body
            jacp[:] = 0
            mujoco.mj_jacBody(self.model, self.data, jacp, None, hand_body_id)

            # Extract columns for our arm joints only
            # qpos_ids map to qvel indices (for 1-DOF hinge joints, dof_id = qpos_id - 7 + freejoint_offset)
            dof_ids = []
            for aid in act_ids:
                jnt_id = self.model.actuator_trnid[aid, 0]
                dof_ids.append(self.model.jnt_dofadr[jnt_id])

            J_arm = jacp[:, dof_ids]  # 3x7 Jacobian for arm joints

            # Damped least-squares: dq = J^T (J J^T + λI)^-1 error
            lam = 0.01
            JJT = J_arm @ J_arm.T + lam * np.eye(3)
            dq = J_arm.T @ np.linalg.solve(JJT, error)

            # Apply joint updates
            for i, qid in enumerate(qpos_ids):
                self.data.qpos[qid] += step_size * dq[i]

        # Read out the solution
        result = tuple(self.data.qpos[qid] for qid in qpos_ids)

        # Restore original qpos (IK was exploratory, don't disturb physics)
        self.data.qpos[:] = orig_qpos
        mujoco.mj_forward(self.model, self.data)

        return result

    def set_hand_grasp(self, hand: str = "right", grasp: bool = True) -> None:
        """Open or close the dexterous hand fingers."""
        self.set_hand_grasp_partial(hand, amount=1.0 if grasp else 0.0)

    def set_hand_grasp_partial(self, hand: str = "right", amount: float = 1.0) -> None:
        """Set hand finger closure amount (0.0 = open, 1.0 = fully closed)."""
        assert self.data is not None
        prefix = "right" if hand == "right" else "left"
        grasp_value = 0.8 * amount
        for finger in ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]:
            idx = G1_ACTUATORS.get(f"{prefix}_{finger}")
            if idx is not None:
                self.data.ctrl[idx] = grasp_value

    def set_hand_fingers(self, hand: str = "right", values: list[float] | None = None) -> None:
        """Set individual finger joint targets (7 values for Dex3-1 hand).

        Order: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
        """
        assert self.data is not None
        if values is None:
            return
        prefix = "right" if hand == "right" else "left"
        fingers = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
        for finger, val in zip(fingers, values):
            idx = G1_ACTUATORS.get(f"{prefix}_{finger}")
            if idx is not None:
                self.data.ctrl[idx] = val

    def get_item_position(self, item_name: str) -> np.ndarray | None:
        """Get the position of a grocery item by name."""
        assert self.model is not None and self.data is not None
        safe_name = item_name.replace(" ", "_")
        body_name = f"item_{safe_name}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return None
        return self.data.xpos[body_id].copy()

    def get_item_info(self, item_name: str) -> dict | None:
        """Get store metadata for an item."""
        return STORE_ITEMS.get(item_name)

    def mark_item_picked(self, item_name: str) -> bool:
        """Mark an item as picked up from the shelf."""
        if item_name in self._items_on_shelf and self._items_on_shelf[item_name]:
            self._items_on_shelf[item_name] = False
            self._items_in_cart.append(item_name)
            self.attach_item_to_hand(item_name)
            return True
        return False

    def move_item_to(self, item_name: str, pos: np.ndarray | list) -> None:
        """Move an item body to a specific world position."""
        assert self.model is not None
        safe_name = item_name.replace(" ", "_")
        body_name = f"item_{safe_name}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            self.model.body_pos[body_id] = pos

    def attach_item_to_hand(self, item_name: str) -> None:
        """Attach an item to the right hand — it will follow the hand each step."""
        assert self.model is not None and self.data is not None
        safe_name = item_name.replace(" ", "_")
        body_name = f"item_{safe_name}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return
        self._held_item_body_id = body_id
        # Snap to hand — takes effect on next mj_step, no mj_forward needed
        hand_pos = self.get_hand_position("right")
        self.model.body_pos[body_id] = hand_pos

    def detach_item(self) -> None:
        """Detach held item and hide it (dropped into cart)."""
        if self._held_item_body_id >= 0:
            # Move to cart basket position so it looks like it landed there
            self.model.body_pos[self._held_item_body_id] = [0.5, -0.3, 0.4]
            self._held_item_body_id = -1

    def _hide_item(self, item_name: str) -> None:
        """Move an item body underground so it disappears from view."""
        assert self.model is not None and self.data is not None
        safe_name = item_name.replace(" ", "_")
        body_name = f"item_{safe_name}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return
        # Move the body's position far below the floor
        self.model.body_pos[body_id][2] = -10.0

    @property
    def items_in_cart(self) -> list[str]:
        return list(self._items_in_cart)

    @property
    def available_items(self) -> list[str]:
        return [k for k, v in self._items_on_shelf.items() if v]

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray | None:
        """Render the current scene to a numpy array (RGB)."""
        if self.model is None or self.data is None:
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def render_robot_view(self, width: int = 640, height: int = 480) -> np.ndarray | None:
        """Render from the robot's perspective — what it 'sees' looking forward.

        Camera is placed at head height, looking in the robot's facing direction.
        Returns RGB numpy array (JPEG-encodable).
        """
        if self.model is None or self.data is None:
            return None
        import math

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()

        # Robot position and heading
        robot_pos = self.get_robot_position()
        # Extract yaw from quaternion qpos[3:7]
        w, x, y, z = self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Camera at head height, looking forward
        head_height = 1.4  # G1 is ~1.3m tall, camera slightly above
        look_dist = 1.5    # Look 1.5m ahead

        camera.lookat[0] = robot_pos[0] + look_dist * math.cos(yaw)
        camera.lookat[1] = robot_pos[1] + look_dist * math.sin(yaw)
        camera.lookat[2] = robot_pos[2] + 0.5  # Look at shelf height

        camera.distance = look_dist
        camera.azimuth = math.degrees(yaw) + 180  # Face forward
        camera.elevation = -15  # Slight downward angle toward shelves

        renderer.update_scene(self.data, camera)
        frame = renderer.render()
        renderer.close()
        return frame

    def render_robot_view_jpeg(self, width: int = 640, height: int = 480) -> bytes | None:
        """Render robot's view as JPEG bytes (ready for VLM)."""
        frame = self.render_robot_view(width, height)
        if frame is None:
            return None
        import cv2
        _, jpeg = cv2.imencode(".jpg", frame[:, :, ::-1])  # RGB → BGR for cv2
        return jpeg.tobytes()

    def reset(self) -> None:
        """Reset the environment to initial state."""
        if self.model is not None:
            # Restore item positions that were hidden
            for item_name, info in STORE_ITEMS.items():
                safe_name = item_name.replace(" ", "_")
                body_name = f"item_{safe_name}"
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id != -1:
                    ix, iy, iz = info["position"]
                    self.model.body_pos[body_id] = [ix, iy, iz]

            self.data = mujoco.MjData(self.model)
            self.data.qpos[2] = G1_INITIAL_HEIGHT
            self._apply_standing_pose()
            mujoco.mj_forward(self.model, self.data)
            # Let robot settle
            for _ in range(500):
                self._apply_standing_pose()
                mujoco.mj_step(self.model, self.data)
        self._items_on_shelf = {k: True for k in STORE_ITEMS}
        self._items_in_cart.clear()
        self._held_item_body_id = -1

    def close(self) -> None:
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._xml_path and os.path.exists(self._xml_path):
            os.unlink(self._xml_path)
            self._xml_path = None
