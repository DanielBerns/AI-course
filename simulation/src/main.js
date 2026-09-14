import { Engine } from './simulation/Engine.js';
import { CanvasRenderer } from './renderer/CanvasRenderer.js';
import { BrainVisualizer } from './ui/BrainVisualizer.js';
import { Charts } from './ui/Charts.js';
import { Controls } from './ui/Controls.js';

document.addEventListener('DOMContentLoaded', () => {
  const simCanvas = document.getElementById('sim-canvas');
  const brainCanvas = document.getElementById('brain-canvas');
  const chartCanvas = document.getElementById('chart-canvas');

  if (!simCanvas || !brainCanvas || !chartCanvas) {
    console.error('Canvas elements missing!');
    return;
  }

  // Set canvas internal dimensions to match display container
  const setCanvasSize = (canvas) => {
    canvas.width = canvas.parentElement.clientWidth || 300;
    canvas.height = canvas.parentElement.clientHeight || 200;
  };

  setCanvasSize(brainCanvas);
  setCanvasSize(chartCanvas);

  // Initialize Simulation Engine & Components
  const engine = new Engine(simCanvas.clientWidth, simCanvas.clientHeight);
  const renderer = new CanvasRenderer(simCanvas, engine);
  const brainVis = new BrainVisualizer(brainCanvas);
  const charts = new Charts(chartCanvas);
  const controls = new Controls(engine, renderer, brainVis);

  window.addEventListener('resize', () => {
    setCanvasSize(brainCanvas);
    setCanvasSize(chartCanvas);
  });

  // 60 FPS Main Loop
  function loop() {
    engine.step();
    renderer.render();
    brainVis.render();
    charts.render(engine);
    controls.updateTelemetry();

    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
});
