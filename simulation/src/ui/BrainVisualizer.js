/**
 * Interactive Neural Network Inspector Canvas Component
 * Renders topology, live activations, and synapse weights for any boid (up to 1000 total neurons).
 */
export class BrainVisualizer {
  /**
   * @param {HTMLCanvasElement} canvas 
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.selectedBoid = null;

    this.nodePositions = []; // Cached node canvas coordinates
  }

  setBoid(boid) {
    this.selectedBoid = boid;
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Clear inspector background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    if (!this.selectedBoid || !this.selectedBoid.brain) {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click any Boid in the simulation to inspect its Brain', w / 2, h / 2);
      return;
    }

    const brain = this.selectedBoid.brain;
    const layerSizes = brain.layerSizes;
    const numLayers = layerSizes.length;

    // Calculate node coordinates per layer
    this.nodePositions = [];
    const paddingX = 50;
    const paddingY = 40;
    const availWidth = w - paddingX * 2;
    const layerStepX = availWidth / (numLayers - 1);

    for (let l = 0; l < numLayers; l++) {
      const numNeurons = layerSizes[l];
      const availHeight = h - paddingY * 2;
      const stepY = numNeurons > 1 ? availHeight / Math.min(numNeurons - 1, 40) : 0;
      
      const layerNodes = [];
      // To keep visual rendering clean for large layers (up to 1000 neurons), sample or compress display vertically
      const displayCount = Math.min(numNeurons, 36);
      const isSubsampled = numNeurons > displayCount;
      const displayStepY = displayCount > 1 ? availHeight / (displayCount - 1) : 0;

      for (let i = 0; i < displayCount; i++) {
        const actualIdx = isSubsampled ? Math.floor((i / (displayCount - 1)) * (numNeurons - 1)) : i;
        const x = paddingX + l * layerStepX;
        const y = paddingY + i * displayStepY;
        layerNodes.push({ x, y, actualIdx });
      }
      this.nodePositions.push(layerNodes);
    }

    // 1. Draw Weights / Synapses
    for (let l = 0; l < brain.weights.length; l++) {
      const weightMatrix = brain.weights[l];
      const fromNodes = this.nodePositions[l];
      const toNodes = this.nodePositions[l + 1];

      for (let i = 0; i < toNodes.length; i++) {
        const toNode = toNodes[i];
        const targetRow = weightMatrix[toNode.actualIdx];
        if (!targetRow) continue;

        for (let j = 0; j < fromNodes.length; j++) {
          const fromNode = fromNodes[j];
          const wVal = targetRow[fromNode.actualIdx] || 0;

          if (Math.abs(wVal) < 0.08) continue; // Skip near-zero connections for speed & clarity

          ctx.beginPath();
          ctx.moveTo(fromNode.x, fromNode.y);
          ctx.lineTo(toNode.x, toNode.y);

          // Color by weight sign: Positive = Cyan/Blue, Negative = Pink/Red
          const alpha = Math.min(0.8, Math.abs(wVal) * 0.4);
          ctx.strokeStyle = wVal > 0 ? `rgba(6, 182, 212, ${alpha})` : `rgba(244, 114, 182, ${alpha})`;
          ctx.lineWidth = Math.min(3, Math.abs(wVal) * 1.5);
          ctx.stroke();
        }
      }
    }

    // 2. Draw Neurons & Activation Signals
    const inputLabels = ['FoodRays', 'PredRays', 'MateRays', 'ShelterRays', 'Flocking', 'Internal'];
    const outputLabels = ['Steer', 'Throttle', 'Sprint', 'Mate', 'Hide', 'Cohesion', 'Separation', 'Signal'];

    for (let l = 0; l < numLayers; l++) {
      const layerNodes = this.nodePositions[l];
      const layerActivations = brain.activations[l];

      for (let i = 0; i < layerNodes.length; i++) {
        const node = layerNodes[i];
        const actVal = layerActivations ? layerActivations[node.actualIdx] : 0;
        const normVal = Math.max(0, Math.min(1, (actVal + 1) / 2)); // Normalize [-1, 1] to [0, 1]

        ctx.save();
        ctx.beginPath();
        const r = l === 0 || l === numLayers - 1 ? 7 : 5;
        ctx.arc(node.x, node.y, r, 0, Math.PI * 2);

        // Node Glow based on activation
        ctx.fillStyle = `rgba(56, 189, 248, ${0.3 + normVal * 0.7})`;
        ctx.shadowColor = normVal > 0.5 ? '#38bdf8' : '#ec4899';
        ctx.shadowBlur = normVal * 12;
        ctx.fill();

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();
      }

      // Layer Title
      ctx.fillStyle = 'rgba(226, 232, 240, 0.7)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      let title = `L${l}`;
      if (l === 0) title = `Input (${layerSizes[0]})`;
      else if (l === numLayers - 1) title = `Output (${layerSizes[l]})`;
      else title = `Hidden ${l} (${layerSizes[l]})`;

      ctx.fillText(title, paddingX + l * layerStepX, h - 10);
    }
  }
}
