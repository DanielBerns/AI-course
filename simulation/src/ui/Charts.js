/**
 * Real-time Telemetry Analytics Chart Renderer
 */
export class Charts {
  /**
   * @param {HTMLCanvasElement} canvas 
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  render(engine) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const history = engine.statsHistory;

    // Clear chart background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    if (history.length < 2) {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.5)';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Collecting Telemetry Data...', w / 2, h / 2);
      return;
    }

    const padding = 25;
    const chartW = w - padding * 2;
    const chartH = h - padding * 2;

    // Draw axes & grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, h - padding);
    ctx.lineTo(w - padding, h - padding);
    ctx.stroke();

    // Scale calculations
    const maxVal = Math.max(
      ...history.map(d => Math.max(d.boidCount, d.foodCount, d.predatorCount * 10))
    , 80);

    const getX = (idx) => padding + (idx / (engine.maxStatsHistory - 1)) * chartW;
    const getY = (val) => h - padding - (val / maxVal) * chartH;

    // 1. Draw Food trend (Emerald)
    this.drawLine(ctx, history, d => d.foodCount, getX, getY, '#10b981', 1.5);

    // 2. Draw Boid trend (Cyan)
    this.drawLine(ctx, history, d => d.boidCount, getX, getY, '#38bdf8', 2.0);

    // 3. Draw Predator trend (Crimson, scaled up 10x for visibility)
    this.drawLine(ctx, history, d => d.predatorCount * 10, getX, getY, '#ef4444', 1.5);

    // Chart Legend
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';

    ctx.fillStyle = '#38bdf8';
    ctx.fillText('● Boids', padding + 10, padding + 12);

    ctx.fillStyle = '#ef4444';
    ctx.fillText('● Predators (x10)', padding + 85, padding + 12);

    ctx.fillStyle = '#10b981';
    ctx.fillText('● Food', padding + 210, padding + 12);
  }

  drawLine(ctx, data, valFn, getX, getY, color, width) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();

    for (let i = 0; i < data.length; i++) {
      const x = getX(i);
      const y = getY(valFn(data[i]));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }
}
