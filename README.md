# Sous Bot 🤖🛒

**An assistive grocery robot for visually impaired and elderly users.**

Sous Bot scans your kitchen, understands your meal plan, figures out what's missing, and helps you shop — powered by a Unitree G1 humanoid robot.

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
│       ├── vision/          # Camera → Vision LLM → Inventory
│       ├── voice/           # STT + TTS accessibility layer
│       ├── planner/         # Meal plan → Shopping list
│       ├── robotics/        # MuJoCo sim / G1 control
│       └── api/             # FastAPI backend
├── sim/
│   └── grocery_env.py       # MuJoCo grocery store environment
└── tests/
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

# Run the voice assistant
uv run python -m sous_bot.voice --text
```

## Team

Built at Nebius.Build SF Hackathon — March 15, 2026
