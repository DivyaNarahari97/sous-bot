# PantryPilot 🤖🛒

**An assistive grocery robot for visually impaired and elderly users.**

PantryPilot scans your kitchen, understands your meal plan, figures out what's missing, and helps you shop — powered by a Unitree G1 humanoid robot.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PantryPilot System                 │
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
pantry-pilot/
├── README.md
├── requirements.txt
├── src/
│   ├── vision/
│   │   └── pantry_scanner.py      # Camera → Vision LLM → Inventory
│   ├── planner/
│   │   └── meal_planner.py        # Meal plan → Shopping list
│   ├── voice/
│   │   └── voice_interface.py     # STT + TTS accessibility layer
│   ├── robot/
│   │   └── grocery_navigator.py   # MuJoCo sim / G1 control
│   └── app.py                     # Main orchestrator
└── sim/
    └── grocery_env.py             # MuJoCo grocery store environment
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/DivyaNarahari97/sous-bot.git
cd sous-bot

# Install dependencies
uv sync

# Copy env file and add your API keys
cp .env.example .env

# Run the app
uv run python src/app.py
```

## Team

Built at Nebius.Build SF Hackathon — March 15, 2026
