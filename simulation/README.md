# Neural Network Boid Ecosystem Simulation 🧬🧠

An interactive, high-performance web simulation of autonomous boids driven by configurable feedforward neural networks (supporting up to 1,000 neurons per brain). Boids evolve through natural selection, learning to forage for food, search for mates to reproduce, seek shelter, and evade apex predators.

---

## 🌟 Features

- **🧠 Up to 1,000 Neuron Brain Architecture**: Customizable feedforward neural network topology ([NeuralNetwork.js](src/brain/NeuralNetwork.js)). Preset configurations range from 136 to 960 total neurons.
- **🧬 Genetic Evolution & Crossover**: Parents pass down neural weights and biases to offspring via uniform crossover and Gaussian mutations ([Genome.js](src/brain/Genome.js)).
- **📡 32 Sensory Input Channels**: Multi-directional raycasts (360° perception for Food, Predators, Mates, Shelters), local flocking metrics (density, alignment, center of mass), and internal vitals (Energy, Libido, Health, Speed, Shelter status).
- **🕹️ 8 Motor Output Actions**: Steer angle, throttle acceleration, sprint boost, mating signal, shelter resting, cohesion bias, separation bias, and signal tone.
- **🛡️ Safe Shelter Domes**: Glassmorphic protective zones where boids hide from predators, reduce energy burn, and regenerate health.
- **🩸 Apex Predators**: Dynamic predators that hunt non-sheltered boids in real time.
- **🔬 Live Brain Inspector**: Interactive canvas rendering node firing activations and color-coded synapse weights for any selected boid.
- **📈 Real-Time Ecosystem Telemetry**: Dynamic line charts plotting prey, predator, food, and fitness evolution over time.
- **🛠️ God-Mode Interactive Tools**: Spawn food, boids, predators, or place custom shelter zones anywhere on the canvas.

---

## 🚀 Quick Start & How to Run

### Prerequisites
- [Node.js](https://nodejs.org/) (v18 or higher recommended)
- `npm` (comes with Node.js)

### Installation & Execution

1. **Clone or Navigate to the project directory**:
   ```bash
   cd /home/dberns/Projects/github/DanielBerns/AI-course/simulation
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Start the Live Development Server**:
   ```bash
   npm run dev
   ```

4. **Open in Browser**:
   Navigate to `http://localhost:5173/` in your web browser.

---

## 🛠️ Build for Production

To create an optimized production bundle:

```bash
npm run build
```

To preview the built production app locally:

```bash
npm run preview
```

---

## 🎮 How to Interact with the Simulation

- **Play / Pause / Speed Control**: Use the top-left HUD to toggle pause or run the simulation at **1x**, **2x**, **5x**, or **10x** speed.
- **Inspect a Boid Brain**: Click the **`🔍 Select Boid`** tool in the bottom HUD, then click any boid on the canvas. The right sidebar will display its live neural network diagram, activations, and telemetry vitals.
- **God-Mode Tools**:
  - `🍃 Spawn Food`: Click anywhere to drop nutrient food items.
  - `🧬 Spawn Boid`: Click to spawn a new neural boid.
  - `🩸 Spawn Predator`: Click to drop an apex predator.
  - `🛡 Place Shelter`: Click to place a safe shelter dome.
- **Brain Topology Configurator**: Click **`🧠 Brain Topology`** in the top HUD to change the neural network layer sizing (up to 1,000 total neurons).

---

## 📂 Project Architecture

```
simulation/
├── index.html                  # Main application HTML markup & UI HUD
├── package.json                # Project dependencies and Vite build scripts
├── src/
│   ├── main.js                 # Application entry point & 60 FPS loop
│   ├── style.css               # Futuristic glassmorphic dark theme styles
│   ├── brain/
│   │   ├── NeuralNetwork.js    # Feedforward network engine (up to 1000 neurons)
│   │   └── Genome.js           # Genetic crossover & mutation utilities
│   ├── simulation/
│   │   ├── Engine.js           # World manager, physics collision, and stats
│   │   ├── Boid.js             # Neural prey boid entity
│   │   ├── Predator.js         # Apex predator hunting AI
│   │   ├── Food.js             # Food nutrient node
│   │   ├── Shelter.js          # Protective shelter dome
│   │   └── Sensors.js          # 32-channel raycast & vital perception engine
│   ├── renderer/
│   │   ├── CanvasRenderer.js   # 60 FPS HTML5 Canvas visual renderer
│   │   └── ParticleSystem.js   # Eating, mating, and attack particle effects
│   └── ui/
│       ├── BrainVisualizer.js  # Live neural network firing canvas
│       ├── Charts.js           # Real-time telemetry line graph
│       └── Controls.js         # Interactive HUD & modal manager
```

---

## 📜 License

MIT License. Free for educational and research use!
