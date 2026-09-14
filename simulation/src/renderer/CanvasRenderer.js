import { ParticleSystem } from './ParticleSystem.js';

export class CanvasRenderer {
  /**
   * @param {HTMLCanvasElement} canvas 
   * @param {import('../simulation/Engine.js').Engine} engine 
   */
  constructor(canvas, engine) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.engine = engine;

    this.particleSystem = new ParticleSystem();
    this.engine.particleSystem = this.particleSystem;

    this.showVisionRays = false;
    this.showVelocityVectors = false;
    this.showShelterZones = true;

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const parent = this.canvas.parentElement;
    if (parent) {
      this.canvas.width = parent.clientWidth || 1200;
      this.canvas.height = parent.clientHeight || 800;
      this.engine.width = this.canvas.width;
      this.engine.height = this.canvas.height;
    }
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Dark Biotech Canvas Background
    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, w, h);

    // Subtle background grid
    this.drawGrid(ctx, w, h);

    // 1. Render Shelters
    if (this.showShelterZones) {
      for (const s of this.engine.shelters) {
        this.drawShelter(ctx, s);
      }
    }

    // 2. Render Food
    for (const food of this.engine.foods) {
      this.drawFood(ctx, food);
    }

    // 3. Render Boids & Trails
    for (const boid of this.engine.boids) {
      this.drawBoidTrail(ctx, boid);
    }
    for (const boid of this.engine.boids) {
      this.drawBoid(ctx, boid);
    }

    // 4. Render Predators
    for (const pred of this.engine.predators) {
      this.drawPredator(ctx, pred);
    }

    // 5. Render Particle System
    this.particleSystem.update();
    this.particleSystem.render(ctx);

    // 6. Selected Boid Overlay & Sensors
    if (this.engine.selectedBoid && !this.engine.selectedBoid.isDead()) {
      this.drawSelectedOverlay(ctx, this.engine.selectedBoid);
    }
  }

  drawGrid(ctx, w, h) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    const gridSize = 50;

    ctx.beginPath();
    for (let x = 0; x < w; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  drawShelter(ctx, shelter) {
    ctx.save();
    const grad = ctx.createRadialGradient(
      shelter.x, shelter.y, shelter.radius * 0.1,
      shelter.x, shelter.y, shelter.radius
    );
    grad.addColorStop(0, 'rgba(16, 185, 129, 0.25)');
    grad.addColorStop(0.7, 'rgba(6, 182, 212, 0.15)');
    grad.addColorStop(1, 'rgba(6, 182, 212, 0.02)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(shelter.x, shelter.y, shelter.radius, 0, Math.PI * 2);
    ctx.fill();

    // Glowing border
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.arc(shelter.x, shelter.y, shelter.radius + Math.sin(shelter.pulse) * 3, 0, Math.PI * 2);
    ctx.stroke();

    // Label
    ctx.fillStyle = 'rgba(6, 182, 212, 0.7)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SAFE SHELTER', shelter.x, shelter.y - shelter.radius - 8);
    ctx.restore();
  }

  drawFood(ctx, food) {
    ctx.save();
    const r = food.radius + Math.sin(food.pulseAngle) * 1.2;

    // Glow
    ctx.shadowColor = '#10b981';
    ctx.shadowBlur = 10;

    ctx.fillStyle = '#34d399';
    ctx.beginPath();
    ctx.arc(food.x, food.y, r, 0, Math.PI * 2);
    ctx.fill();

    // Core
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(food.x, food.y, r * 0.4, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  drawBoidTrail(ctx, boid) {
    if (boid.trail.length < 2) return;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(boid.trail[0].x, boid.trail[0].y);

    for (let i = 1; i < boid.trail.length; i++) {
      ctx.lineTo(boid.trail[i].x, boid.trail[i].y);
    }

    const strokeColor = boid.gender === 'male' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(244, 114, 182, 0.25)';
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = boid.isSprinting ? 2.5 : 1.2;
    ctx.stroke();
    ctx.restore();
  }

  drawBoid(ctx, boid) {
    ctx.save();
    ctx.translate(boid.x, boid.y);
    ctx.rotate(boid.heading);

    const isMale = boid.gender === 'male';
    const mainColor = isMale ? '#38bdf8' : '#f472b6';
    const accentColor = isMale ? '#0284c7' : '#db2777';

    // Highlight selected or sprinting
    if (boid.isSelected) {
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 12;
    } else if (boid.isSprinting) {
      ctx.shadowColor = '#f97316';
      ctx.shadowBlur = 10;
    }

    // Body Triangle
    ctx.fillStyle = mainColor;
    ctx.beginPath();
    ctx.moveTo(boid.radius * 1.6, 0);
    ctx.lineTo(-boid.radius * 1.2, -boid.radius * 0.9);
    ctx.lineTo(-boid.radius * 0.6, 0);
    ctx.lineTo(-boid.radius * 1.2, boid.radius * 0.9);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Energy Bar Arc
    const energyPct = boid.energy / boid.maxEnergy;
    ctx.rotate(-boid.heading); // Un-rotate for bar

    ctx.beginPath();
    ctx.arc(0, 0, boid.radius + 4, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * energyPct));
    ctx.strokeStyle = energyPct > 0.3 ? '#10b981' : '#ef4444';
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // Icons / Indicators
    if (boid.isMatingSignaling) {
      ctx.fillStyle = '#ec4899';
      ctx.font = '10px sans-serif';
      ctx.fillText('♥', -3, -boid.radius - 8);
    } else if (boid.inShelter) {
      ctx.fillStyle = '#06b6d4';
      ctx.font = '9px sans-serif';
      ctx.fillText('🛡', -3, -boid.radius - 8);
    }

    ctx.restore();
  }

  drawPredator(ctx, pred) {
    ctx.save();
    ctx.translate(pred.x, pred.y);
    ctx.rotate(pred.heading);

    // Red Menacing Vision Cone
    ctx.fillStyle = 'rgba(239, 68, 68, 0.06)';
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, pred.visionRadius * 0.6, -Math.PI / 4, Math.PI / 4);
    ctx.closePath();
    ctx.fill();

    // Predator Body (Aggressive Diamond Star)
    ctx.shadowColor = '#ef4444';
    ctx.shadowBlur = 14;
    ctx.fillStyle = '#dc2626';

    ctx.beginPath();
    ctx.moveTo(pred.radius * 1.8, 0);
    ctx.lineTo(-pred.radius * 1.2, -pred.radius * 1.1);
    ctx.lineTo(-pred.radius * 0.4, 0);
    ctx.lineTo(-pred.radius * 1.2, pred.radius * 1.1);
    ctx.closePath();
    ctx.fill();

    // Eye
    ctx.fillStyle = '#fef08a';
    ctx.beginPath();
    ctx.arc(pred.radius * 0.6, 0, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.rotate(-pred.heading);

    // Energy / Target indicator line
    if (pred.targetBoid) {
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(pred.targetBoid.x - pred.x, pred.targetBoid.y - pred.y);
      ctx.stroke();
    }

    ctx.restore();
  }

  drawSelectedOverlay(ctx, boid) {
    ctx.save();

    // Target Reticle around selected boid
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(boid.x, boid.y, boid.radius + 12, 0, Math.PI * 2);
    ctx.stroke();

    // Vision Radius Circle
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.25)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(boid.x, boid.y, boid.visionRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw Sensory Perception Rays
    ctx.setLineDash([]);
    const rayCount = 8;
    const angleStep = (Math.PI * 2) / rayCount;

    for (let i = 0; i < rayCount; i++) {
      const angle = boid.heading + (i * angleStep);
      const rx = boid.x + Math.cos(angle) * boid.visionRadius;
      const ry = boid.y + Math.sin(angle) * boid.visionRadius;

      ctx.strokeStyle = 'rgba(251, 191, 36, 0.15)';
      ctx.beginPath();
      ctx.moveTo(boid.x, boid.y);
      ctx.lineTo(rx, ry);
      ctx.stroke();
    }

    // Telemetry label
    ctx.fillStyle = '#fbbf24';
    ctx.font = '11px monospace';
    ctx.fillText(`BOID #${boid.id} [Gen ${boid.generation}]`, boid.x - 40, boid.y + boid.radius + 24);

    ctx.restore();
  }
}
