export class Predator {
  static idCounter = 0;

  constructor(x = 0, y = 0) {
    this.id = ++Predator.idCounter;
    this.x = x;
    this.y = y;
    this.velocity = {
      x: (Math.random() - 0.5) * 3,
      y: (Math.random() - 0.5) * 3
    };
    this.heading = Math.atan2(this.velocity.y, this.velocity.x);

    this.radius = 11;
    this.maxSpeed = 4.2;
    this.visionRadius = 220;
    
    this.maxEnergy = 150;
    this.energy = 100;
    this.kills = 0;
    this.age = 0;
    this.state = 'wandering'; // 'wandering', 'chasing', 'eating'
    this.targetBoid = null;
    this.trail = [];
  }

  update(engine) {
    this.age++;
    this.energy -= 0.08; // Metabolism

    // 1. Search for nearest target boid NOT in shelter
    let nearestDist = Infinity;
    let closestBoid = null;

    for (const boid of engine.boids) {
      if (boid.inShelter) continue; // Shelters protect boids!

      const dx = boid.x - this.x;
      const dy = boid.y - this.y;
      const dist = Math.hypot(dx, dy);

      if (dist < this.visionRadius && dist < nearestDist) {
        nearestDist = dist;
        closestBoid = boid;
      }
    }

    this.targetBoid = closestBoid;

    if (this.targetBoid) {
      this.state = 'chasing';
      // Pursue target boid
      const dx = this.targetBoid.x - this.x;
      const dy = this.targetBoid.y - this.y;
      const angle = Math.atan2(dy, dx);

      const targetVx = Math.cos(angle) * this.maxSpeed;
      const targetVy = Math.sin(angle) * this.maxSpeed;

      this.velocity.x += (targetVx - this.velocity.x) * 0.12;
      this.velocity.y += (targetVy - this.velocity.y) * 0.12;

      // Check kill / eating radius
      if (nearestDist <= this.radius + this.targetBoid.radius) {
        // Attack & Consume Boid
        this.targetBoid.health = 0; // Kills boid
        this.energy = Math.min(this.maxEnergy, this.energy + 60);
        this.kills++;
        this.state = 'eating';

        // Spawn particle alert
        if (engine.particleSystem) {
          engine.particleSystem.spawnAttackEffect(this.x, this.y);
        }
      }
    } else {
      this.state = 'wandering';
      // Wander around smoothly with subtle steering noise
      const angle = this.heading + (Math.random() - 0.5) * 0.3;
      const wanderVx = Math.cos(angle) * (this.maxSpeed * 0.6);
      const wanderVy = Math.sin(angle) * (this.maxSpeed * 0.6);

      this.velocity.x += (wanderVx - this.velocity.x) * 0.05;
      this.velocity.y += (wanderVy - this.velocity.y) * 0.05;
    }

    // Apply movement
    this.x += this.velocity.x;
    this.y += this.velocity.y;
    this.heading = Math.atan2(this.velocity.y, this.velocity.x);

    // Boundary Bounce
    const margin = 25;
    if (this.x < margin) { this.x = margin; this.velocity.x *= -1; }
    if (this.x > engine.width - margin) { this.x = engine.width - margin; this.velocity.x *= -1; }
    if (this.y < margin) { this.y = margin; this.velocity.y *= -1; }
    if (this.y > engine.height - margin) { this.y = engine.height - margin; this.velocity.y *= -1; }

    // Trail history
    if (engine.tickCount % 4 === 0) {
      this.trail.push({ x: this.x, y: this.y });
      if (this.trail.length > 6) this.trail.shift();
    }
  }

  isDead() {
    return this.energy <= 0;
  }
}
