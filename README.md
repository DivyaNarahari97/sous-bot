# Sous Bot 🤖🛒

**An assistive grocery robot for visually impaired and elderly users.**

Sous Bot scans your kitchen, understands your meal plan, figures out what's missing, and helps you shop — powered by a Unitree G1 humanoid robot.

![Sous Bot Simulation Demo](demo.png)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Sous Bot System                    │
├─────────────┬──────────────┬────────────────────────┤
│  PERCEIVE   │    REASON    │         ACT            │
│             │              │                        │
│ Camera Feed │ Meal Planner │ MuJoCo Sim / G1 Robot  │
│     ↓       │ (LLM Agent)  │                        │
│ Vision LLM  │     ↓        │ - Navigate grocery     │
│ (Nebius TF) │ Shopping     │ - Locate items         │
│     ↓       │ List Gen     │ - Reach & grasp        │
│ Pantry      │              │ - Hand off to user     │
│ Inventory   │              │                        │
├─────────────┴──────────────┴────────────────────────┤
│              VOICE INTERFACE (Accessibility)         │
│         Speech-to-Text ←→ Text-to-Speech            │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

- **Vision**: Nebius Token Factory (Qwen2.5-VL / LLaVA) for pantry scanning
- **Reasoning**: Nebius Token Factory LLM for meal planning + list generation
- **Voice**: Whisper (STT) + TTS for accessibility
- **Simulation**: MuJoCo with Unitree G1 model
- **Robotics**: Navigation + reach-and-grasp policies
- **Backend**: Python, FastAPI

## Project Structure

```
sous-bot/
├── README.md
├── pyproject.toml
├── src/
│   └── sous_bot/
│       ├── vision/          # VLM-based shelf scanning (Qwen2.5-VL)
│       ├── voice/           # STT + TTS accessibility layer
│       ├── planner/         # Meal plan → Shopping list (LLM)
│       ├── robotics/        # Robot adapter + controller
│       │   └── adapters/    # Simulation & hardware adapters
│       └── api/             # FastAPI backend
├── sim/
│   ├── grocery_env.py       # MuJoCo grocery store environment
│   ├── download_textures.py # Product image downloader
│   └── textures/            # 100+ product images (128x128 PNG)
├── scripts/
│   └── run_viewer.py        # Interactive 3D viewer + full pipeline
└── tests/
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Nebius API key (for VLM vision + LLM planner)

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/DivyaNarahari97/sous-bot.git
cd sous-bot

# Install dependencies
uv sync

# Copy env file and add your Nebius API key
cp .env.example .env
```

### Run the Interactive Grocery Simulation

```bash
# Full pipeline: recipe → shopping list → robot fetches items
uv run python scripts/run_viewer.py

# Specify recipes directly
uv run python scripts/run_viewer.py --items carbonara "stir fry"

# Disable VLM vision (use hardcoded item positions)
uv run python scripts/run_viewer.py --no-vision
```

### Run the Voice Assistant

```bash
uv run python -m sous_bot.voice --text
```

## MuJoCo Grocery Simulation

The simulation features a fully stocked grocery store with a **Unitree G1 humanoid robot** (29-DOF + dexterous hands).

**Store Layout:**
- 6 aisles (produce, dairy, bakery, deli, spices, frozen) with 3 shelves each
- 140+ grocery items with real product image textures
- Shopping cart for item collection

**Robot Capabilities:**
- Autonomous navigation with aisle-aware path planning
- Inverse kinematics (damped least-squares) for arm reaching
- Dexterous hand control (7 finger joints per hand) for grasping
- First-person RGBD camera for VLM-based shelf scanning

**Viewer Controls:**
| Key | Action |
|-----|--------|
| SPACE | Start/restart shopping |
| Left-click drag | Rotate camera |
| Right-click drag | Pan camera |
| Scroll | Zoom |
| R | Reset |
| Q / ESC | Quit |

## VLM Vision Pipeline

Uses **Nebius Token Factory Qwen2.5-VL** for:
- **Shelf scanning:** Detects all visible products from the robot's camera feed
- **Item localization:** Finds specific items with pixel coordinates, converted to 3D world positions via depth rendering

The vision pipeline integrates with the robot's first-person RGBD camera to enable vision-guided reaching and grasping.

## Team

Built at Nebius.Build SF Hackathon — March 15, 2026
