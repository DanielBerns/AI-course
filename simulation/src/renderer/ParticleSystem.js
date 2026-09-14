/**
 * Visual Particle Engine for Ecosystem Feedback
 */
export class ParticleSystem {
  constructor() {
    this.particles = [];
  }

  spawnEatEffect(x, y) {
    for (let i = 0; i < 10; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.0 + Math.random() * 2.5;
      this.particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        color: '#10b981', // Emerald green spark
        size: 2.5 + Math.random() * 2,
        life: 1.0,
        decay: 0.04 + Math.random() * 0.03
      });
    }
  }

  spawnMatingEffect(x, y) {
    for (let i = 0; i < 14; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 0.5 + Math.random() * 2.0;
      this.particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 0.5,
        color: '#ec4899', // Pink heart star
        size: 3 + Math.random() * 3,
        life: 1.0,
        decay: 0.02 + Math.random() * 0.02,
        isHeart: true
      });
    }
  }

  spawnAttackEffect(x, y) {
    for (let i = 0; i < 16; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 2.0 + Math.random() * 4.0;
      this.particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        color: '#ef4444', // Crimson flash
        size: 3 + Math.random() * 3,
        life: 1.0,
        decay: 0.05
      });
    }
  }

  update() {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= p.decay;

      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }
  }

  render(ctx) {
    ctx.save();
    for (const p of this.particles) {
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}
