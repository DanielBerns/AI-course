import { NeuralNetwork } from '../brain/NeuralNetwork.js';
import { Sensors } from './Sensors.js';

export class Boid {
  static idCounter = 0;

  /**
   * @param {number} x 
   * @param {number} y 
   * @param {number[]} layerSizes Neural network architecture (up to 1000 total neurons)
   */
  constructor(x = 0, y = 0, layerSizes = [32, 64, 32, 8]) {
    this.id = ++Boid.idCounter;
    this.x = x;
    this.y = y;
    this.velocity = {
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2
    };
    this.acceleration = { x: 0, y: 0 };
    this.heading = Math.atan2(this.velocity.y, this.velocity.x);

    // Physical attributes
    this.radius = 6;
    this.maxSpeed = 3.5;
    this.sprintSpeed = 6.0;
    this.maxForce = 0.2;
    this.visionRadius = 140;

    // Vital Stats
    this.maxEnergy = 100;
    this.energy = 80 + Math.random() * 20;
    this.maxHealth = 100;
    this.health = 100;
    this.maxLibido = 100;
    this.libido = Math.random() * 30; // Mating desire
    this.age = 0;
    this.maxAge = 1800 + Math.floor(Math.random() * 600); // Ticks of life
    this.generation = 1;
    this.gender = Math.random() < 0.5 ? 'male' : 'female';
    this.fitness = 0;
    this.offspringCount = 0;
    this.foodEaten = 0;

    // States
    this.inShelter = false;
    this.isSprinting = false;
    this.isMatingSignaling = false;
    this.isHiding = false;

    // Neural Network Brain
    this.layerSizes = layerSizes;
    this.brain = new NeuralNetwork(layerSizes);
    this.lastNeuralOutputs = new Float32Array(8);

    // Visual trail & selection
    this.trail = [];
    this.isSelected = false;
  }

  isReadyToMate() {
    return this.energy > 45 && this.libido >= 70 && this.age > 100;
  }

  update(engine) {
    this.age++;
    this.fitness += 0.1; // Baseline survival reward

    // 1. Check Shelter status
    this.inShelter = engine.shelters.some(s => s.containsPoint(this.x, this.y));

    // 2. Query Sensors & Execute Neural Network
    const inputs = Sensors.getInputs(this, engine);
    const outputs = this.brain.predict(inputs);
    this.lastNeuralOutputs = outputs;

    // Parse Neural Outputs:
    // [0]: Steer Angle (-1 to 1 -> -PI/4 to PI/4)
    // [1]: Throttle (-1 to 1 -> brake to full acceleration)
    // [2]: Sprint Boost (> 0.2 activates sprint)
    // [3]: Want To Mate (> 0.3 signals mating readiness)
    // [4]: Want To Hide (> 0.3 reduces energy burn in shelter)
    // [5]: Cohesion Bias
    // [6]: Separation Bias
    // [7]: Signal Emitter

    const steerOutput = outputs[0];
    const throttleOutput = outputs[1];
    const sprintOutput = outputs[2];
    const mateOutput = outputs[3];
    const hideOutput = outputs[4];

    this.isSprinting = (sprintOutput > 0.2 && this.energy > 15);
    this.isMatingSignaling = (mateOutput > 0.3 && this.isReadyToMate());
    this.isHiding = (hideOutput > 0.3 && this.inShelter);

    // Apply Steering & Acceleration based on neural decision
    const currentMaxSpeed = this.isSprinting ? this.sprintSpeed : this.maxSpeed;
    const steerAngle = steerOutput * (Math.PI / 3);
    const targetHeading = this.heading + steerAngle;
    
    const targetSpeed = Math.max(0.5, (throttleOutput + 1) / 2) * currentMaxSpeed;
    const targetVx = Math.cos(targetHeading) * targetSpeed;
    const targetVy = Math.sin(targetHeading) * targetSpeed;

    // Steering force towards target velocity
    const steerX = (targetVx - this.velocity.x) * 0.15;
    const steerY = (targetVy - this.velocity.y) * 0.15;

    this.acceleration.x += steerX;
    this.acceleration.y += steerY;

    // Update Physics
    this.velocity.x += this.acceleration.x;
    this.velocity.y += this.acceleration.y;

    // Speed Cap
    const currentSpeed = Math.hypot(this.velocity.x, this.velocity.y);
    if (currentSpeed > currentMaxSpeed) {
      this.velocity.x = (this.velocity.x / currentSpeed) * currentMaxSpeed;
      this.velocity.y = (this.velocity.y / currentSpeed) * currentMaxSpeed;
    }

    this.x += this.velocity.x;
    this.y += this.velocity.y;
    this.heading = Math.atan2(this.velocity.y, this.velocity.x);

    // Reset acceleration for next frame
    this.acceleration.x = 0;
    this.acceleration.y = 0;

    // Screen Boundary Bounce / Wrap
    const margin = 20;
    if (this.x < margin) { this.x = margin; this.velocity.x *= -1; }
    if (this.x > engine.width - margin) { this.x = engine.width - margin; this.velocity.x *= -1; }
    if (this.y < margin) { this.y = margin; this.velocity.y *= -1; }
    if (this.y > engine.height - margin) { this.y = engine.height - margin; this.velocity.y *= -1; }

    // Energy Metabolism
    let energyBurn = 0.05 + (currentSpeed / this.maxSpeed) * 0.08;
    if (this.isSprinting) energyBurn *= 2.5;
    if (this.isHiding && this.inShelter) energyBurn *= 0.3; // Resting bonus in shelter

    this.energy -= energyBurn;
    if (this.energy < 0) {
      this.energy = 0;
      this.health -= 0.5; // Starvation damage
    }

    // Libido growth over time
    if (this.energy > 40 && this.age > 80) {
      this.libido = Math.min(this.maxLibido, this.libido + 0.12);
    }

    // Shelter regeneration bonus
    if (this.inShelter) {
      this.health = Math.min(this.maxHealth, this.health + 0.2);
    }

    // Trail history for aesthetic rendering
    if (engine.tickCount % 3 === 0) {
      this.trail.push({ x: this.x, y: this.y });
      if (this.trail.length > 8) this.trail.shift();
    }
  }

  isDead() {
    return this.health <= 0 || this.age >= this.maxAge;
  }
}
