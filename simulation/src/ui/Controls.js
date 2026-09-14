import { Food } from '../simulation/Food.js';
import { Boid } from '../simulation/Boid.js';
import { Predator } from '../simulation/Predator.js';
import { Shelter } from '../simulation/Shelter.js';

/**
 * Controls HUD Manager
 */
export class Controls {
  /**
   * @param {import('../simulation/Engine.js').Engine} engine 
   * @param {import('../renderer/CanvasRenderer.js').CanvasRenderer} renderer 
   * @param {import('./BrainVisualizer.js').BrainVisualizer} brainVis 
   */
  constructor(engine, renderer, brainVis) {
    this.engine = engine;
    this.renderer = renderer;
    this.brainVis = brainVis;

    this.activeGodTool = 'select'; // 'select', 'food', 'boid', 'predator', 'shelter'

    this.setupEventListeners();
  }

  setupEventListeners() {
    // 1. Play / Pause
    const btnPlayPause = document.getElementById('btn-play-pause');
    if (btnPlayPause) {
      btnPlayPause.addEventListener('click', () => {
        this.engine.isPaused = !this.engine.isPaused;
        btnPlayPause.textContent = this.engine.isPaused ? '▶ Play' : '⏸ Pause';
        btnPlayPause.classList.toggle('active', this.engine.isPaused);
      });
    }

    // 2. Speed multipliers
    const speedBtns = document.querySelectorAll('.btn-speed');
    speedBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const speed = parseInt(btn.dataset.speed || '1', 10);
        this.engine.speedMultiplier = speed;
      });
    });

    // 3. Reset Simulation
    const btnReset = document.getElementById('btn-reset');
    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this.engine.initDefaultWorld();
        this.brainVis.setBoid(null);
      });
    }

    // 4. Sliders
    const inputMutation = document.getElementById('input-mutation-rate');
    if (inputMutation) {
      inputMutation.addEventListener('input', (e) => {
        this.engine.mutationRate = parseFloat(e.target.value);
        document.getElementById('lbl-mutation-rate').textContent = `${(this.engine.mutationRate * 100).toFixed(0)}%`;
      });
    }

    const inputTargetBoids = document.getElementById('input-target-boids');
    if (inputTargetBoids) {
      inputTargetBoids.addEventListener('input', (e) => {
        this.engine.targetBoidCount = parseInt(e.target.value, 10);
        document.getElementById('lbl-target-boids').textContent = this.engine.targetBoidCount;
      });
    }

    const inputTargetPredators = document.getElementById('input-target-predators');
    if (inputTargetPredators) {
      inputTargetPredators.addEventListener('input', (e) => {
        this.engine.targetPredatorCount = parseInt(e.target.value, 10);
        document.getElementById('lbl-target-predators').textContent = this.engine.targetPredatorCount;
      });
    }

    // 5. God Mode Tools
    const toolBtns = document.querySelectorAll('.btn-god-tool');
    toolBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        toolBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeGodTool = btn.dataset.tool;
      });
    });

    // Canvas click event for tool action
    this.renderer.canvas.addEventListener('click', (e) => {
      const rect = this.renderer.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (this.activeGodTool === 'select') {
        const boid = this.engine.selectBoidAt(x, y);
        this.brainVis.setBoid(boid);
      } else if (this.activeGodTool === 'food') {
        this.engine.foods.push(new Food(x, y, 40));
      } else if (this.activeGodTool === 'boid') {
        this.engine.boids.push(new Boid(x, y, this.engine.brainLayerSizes));
      } else if (this.activeGodTool === 'predator') {
        this.engine.predators.push(new Predator(x, y));
      } else if (this.activeGodTool === 'shelter') {
        this.engine.shelters.push(new Shelter(x, y, 80));
      }
    });

    // 6. Brain Architecture Configurator Modal
    const btnBrainConfig = document.getElementById('btn-brain-config');
    const modalBrain = document.getElementById('modal-brain-config');
    const btnSaveBrain = document.getElementById('btn-save-brain-config');
    const btnCloseBrain = document.getElementById('btn-close-brain-config');

    if (btnBrainConfig && modalBrain) {
      btnBrainConfig.addEventListener('click', () => modalBrain.classList.remove('hidden'));
    }
    if (btnCloseBrain && modalBrain) {
      btnCloseBrain.addEventListener('click', () => modalBrain.classList.add('hidden'));
    }

    if (btnSaveBrain) {
      btnSaveBrain.addEventListener('click', () => {
        const inputLayersStr = document.getElementById('input-layer-config').value;
        try {
          const layerSizes = inputLayersStr.split(',').map(s => parseInt(s.trim(), 10));
          if (layerSizes.some(isNaN) || layerSizes.length < 2) {
            alert('Invalid layer sizes. Please enter comma-separated numbers (e.g. 32, 64, 32, 8).');
            return;
          }
          if (layerSizes[0] !== 32) {
            alert('Input layer size must be 32 to match boid sensory engine inputs.');
            return;
          }
          if (layerSizes[layerSizes.length - 1] !== 8) {
            alert('Output layer size must be 8 to match boid motor controls.');
            return;
          }
          const success = this.engine.setBrainArchitecture(layerSizes);
          if (success) {
            modalBrain.classList.add('hidden');
            this.brainVis.setBoid(null);
          }
        } catch (err) {
          alert('Error parsing brain configuration: ' + err.message);
        }
      });
    }
  }

  updateTelemetry() {
    // Update top header telemetry
    const elBoids = document.getElementById('stat-boids');
    const elPredators = document.getElementById('stat-predators');
    const elFood = document.getElementById('stat-food');
    const elGen = document.getElementById('stat-avg-gen');
    const elFitness = document.getElementById('stat-avg-fitness');

    if (elBoids) elBoids.textContent = this.engine.boids.length;
    if (elPredators) elPredators.textContent = this.engine.predators.length;
    if (elFood) elFood.textContent = this.engine.foods.length;

    if (this.engine.statsHistory.length > 0) {
      const latest = this.engine.statsHistory[this.engine.statsHistory.length - 1];
      if (elGen) elGen.textContent = latest.avgGen;
      if (elFitness) elFitness.textContent = latest.avgFitness;
    }

    // Update Selected Boid Panel telemetry
    const panelSelected = document.getElementById('selected-boid-telemetry');
    if (panelSelected) {
      const boid = this.engine.selectedBoid;
      if (boid && !boid.isDead()) {
        panelSelected.innerHTML = `
          <div class="telemetry-row"><span>ID:</span> <strong>#${boid.id} (${boid.gender.toUpperCase()})</strong></div>
          <div class="telemetry-row"><span>Gen / Age:</span> <strong>Gen ${boid.generation} / ${boid.age}t</strong></div>
          <div class="telemetry-row"><span>Energy:</span> <strong>${Math.round(boid.energy)}%</strong></div>
          <div class="telemetry-row"><span>Libido:</span> <strong>${Math.round(boid.libido)}%</strong></div>
          <div class="telemetry-row"><span>Fitness:</span> <strong>${Math.round(boid.fitness)}</strong></div>
          <div class="telemetry-row"><span>Offspring:</span> <strong>${boid.offspringCount}</strong></div>
          <div class="telemetry-row"><span>State:</span> <strong>${boid.inShelter ? '🛡 Sheltered' : (boid.isSprinting ? '⚡ Sprinting' : (boid.isMatingSignaling ? '♥ Mating' : ' Waving'))}</strong></div>
        `;
      } else {
        panelSelected.innerHTML = `<p class="placeholder-text">No Boid Selected</p>`;
      }
    }
  }
}
