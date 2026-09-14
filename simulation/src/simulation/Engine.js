import { Boid } from './Boid.js';
import { Predator } from './Predator.js';
import { Food } from './Food.js';
import { Shelter } from './Shelter.js';
import { Genome } from '../brain/Genome.js';

export class Engine {
  constructor(width = 1200, height = 800) {
    this.width = width;
    this.height = height;

    this.boids = [];
    this.predators = [];
    this.foods = [];
    this.shelters = [];

    this.tickCount = 0;
    this.speedMultiplier = 1;
    this.isPaused = false;

    // Ecosystem Parameters (tweakable via UI)
    this.targetBoidCount = 45;
    this.targetPredatorCount = 3;
    this.targetFoodCount = 60;
    this.mutationRate = 0.05;
    this.mutationAmount = 0.2;
    this.brainLayerSizes = [32, 64, 32, 8]; // Default architecture

    this.selectedBoid = null;
    this.particleSystem = null; // Attached by renderer

    // Analytical metrics history
    this.statsHistory = [];
    this.maxStatsHistory = 200;

    this.initDefaultWorld();
  }

  initDefaultWorld() {
    this.boids = [];
    this.predators = [];
    this.foods = [];
    this.shelters = [];
    this.tickCount = 0;

    // Spawn Shelters
    this.shelters.push(new Shelter(this.width * 0.25, this.height * 0.3, 85));
    this.shelters.push(new Shelter(this.width * 0.75, this.height * 0.7, 95));

    // Initial Boid Population
    for (let i = 0; i < this.targetBoidCount; i++) {
      const x = Math.random() * (this.width - 100) + 50;
      const y = Math.random() * (this.height - 100) + 50;
      this.boids.push(new Boid(x, y, this.brainLayerSizes));
    }

    // Initial Predators
    for (let i = 0; i < this.targetPredatorCount; i++) {
      const x = Math.random() * (this.width - 100) + 50;
      const y = Math.random() * (this.height - 100) + 50;
      this.predators.push(new Predator(x, y));
    }

    // Initial Food
    for (let i = 0; i < this.targetFoodCount; i++) {
      this.spawnRandomFood();
    }
  }

  spawnRandomFood() {
    // Spawn food with slight clustering bias around shelters/edges
    const x = Math.random() * (this.width - 80) + 40;
    const y = Math.random() * (this.height - 80) + 40;
    this.foods.push(new Food(x, y, 35));
  }

  step() {
    if (this.isPaused) return;

    for (let subStep = 0; subStep < this.speedMultiplier; subStep++) {
      this.tickCount++;

      // 1. Update Shelters & Foods
      for (const s of this.shelters) s.update();
      for (const f of this.foods) f.update();

      // 2. Replenish Food dynamically
      if (this.foods.length < this.targetFoodCount && Math.random() < 0.25) {
        this.spawnRandomFood();
      }

      // 3. Update Boids
      for (let i = this.boids.length - 1; i >= 0; i--) {
        const boid = this.boids[i];
        boid.update(this);

        // Check for Food Consumption
        for (let fIdx = this.foods.length - 1; fIdx >= 0; fIdx--) {
          const food = this.foods[fIdx];
          const dx = food.x - boid.x;
          const dy = food.y - boid.y;
          if (dx * dx + dy * dy <= (boid.radius + food.radius) * (boid.radius + food.radius)) {
            // Eat Food
            boid.energy = Math.min(boid.maxEnergy, boid.energy + food.energyValue);
            boid.health = Math.min(boid.maxHealth, boid.health + 10);
            boid.fitness += 15;
            boid.foodEaten++;
            this.foods.splice(fIdx, 1);

            if (this.particleSystem) {
              this.particleSystem.spawnEatEffect(food.x, food.y);
            }
            break;
          }
        }

        // Check for Mating & Reproduction
        if (boid.isReadyToMate() && boid.isMatingSignaling) {
          for (let j = 0; j < this.boids.length; j++) {
            const partner = this.boids[j];
            if (partner === boid) continue;

            if (partner.isReadyToMate() && partner.isMatingSignaling && partner.gender !== boid.gender) {
              const dx = partner.x - boid.x;
              const dy = partner.y - boid.y;
              if (dx * dx + dy * dy <= 22 * 22) {
                // Execute Mating Event!
                this.reproduce(boid, partner);
                break;
              }
            }
          }
        }

        // Remove dead boids
        if (boid.isDead()) {
          if (boid === this.selectedBoid) {
            this.selectedBoid = null;
          }
          this.boids.splice(i, 1);
        }
      }

      // 4. Update Predators
      for (let i = this.predators.length - 1; i >= 0; i--) {
        const predator = this.predators[i];
        predator.update(this);

        if (predator.isDead()) {
          this.predators.splice(i, 1);
        }
      }

      // Maintain minimum populations if depleted
      if (this.boids.length < 8) {
        // Evolutionary rescue: spawn child from top surviving boid or random
        this.spawnEvolutionaryBoid();
      }

      if (this.predators.length < Math.min(1, this.targetPredatorCount) && Math.random() < 0.05) {
        this.predators.push(new Predator(Math.random() * this.width, Math.random() * this.height));
      }

      // Record Stats periodic window
      if (this.tickCount % 30 === 0) {
        this.recordStats();
      }
    }
  }

  reproduce(parentA, parentB) {
    // Consume parent resources
    parentA.energy -= 35;
    parentA.libido = 0;
    parentA.fitness += 50;
    parentA.offspringCount++;

    parentB.energy -= 35;
    parentB.libido = 0;
    parentB.fitness += 50;
    parentB.offspringCount++;

    // Create child network with crossover
    const childBrain = Genome.crossover(
      parentA.brain,
      parentB.brain,
      this.mutationRate,
      this.mutationAmount
    );

    const childX = (parentA.x + parentB.x) / 2 + (Math.random() - 0.5) * 10;
    const childY = (parentA.y + parentB.y) / 2 + (Math.random() - 0.5) * 10;

    const child = new Boid(childX, childY, this.brainLayerSizes);
    child.brain = childBrain;
    child.generation = Math.max(parentA.generation, parentB.generation) + 1;

    this.boids.push(child);

    if (this.particleSystem) {
      this.particleSystem.spawnMatingEffect(childX, childY);
    }
  }

  spawnEvolutionaryBoid() {
    let parent = null;
    if (this.boids.length > 0) {
      // Pick highest fitness survivor
      parent = this.boids.reduce((best, b) => (b.fitness > best.fitness ? b : best), this.boids[0]);
    }

    const x = Math.random() * (this.width - 100) + 50;
    const y = Math.random() * (this.height - 100) + 50;
    const child = new Boid(x, y, this.brainLayerSizes);

    if (parent) {
      child.brain = Genome.cloneAndMutate(parent.brain, this.mutationRate, this.mutationAmount);
      child.generation = parent.generation + 1;
    }
    this.boids.push(child);
  }

  recordStats() {
    const avgFitness = this.boids.length > 0
      ? this.boids.reduce((sum, b) => sum + b.fitness, 0) / this.boids.length
      : 0;
    
    const avgGen = this.boids.length > 0
      ? this.boids.reduce((sum, b) => sum + b.generation, 0) / this.boids.length
      : 1;

    this.statsHistory.push({
      tick: this.tickCount,
      boidCount: this.boids.length,
      predatorCount: this.predators.length,
      foodCount: this.foods.length,
      avgFitness: Math.round(avgFitness),
      avgGen: parseFloat(avgGen.toFixed(1))
    });

    if (this.statsHistory.length > this.maxStatsHistory) {
      this.statsHistory.shift();
    }
  }

  selectBoidAt(x, y) {
    let closest = null;
    let minDist = 25;

    for (const boid of this.boids) {
      const dx = boid.x - x;
      const dy = boid.y - y;
      const dist = Math.hypot(dx, dy);

      if (dist < minDist) {
        minDist = dist;
        closest = boid;
      }
    }

    if (this.selectedBoid) this.selectedBoid.isSelected = false;
    this.selectedBoid = closest;
    if (this.selectedBoid) this.selectedBoid.isSelected = true;

    return this.selectedBoid;
  }

  setBrainArchitecture(layerSizes) {
    const total = layerSizes.reduce((s, n) => s + n, 0);
    if (total > 1000) {
      alert(`Requested architecture exceeds 1000 total neurons (${total}).`);
      return false;
    }
    this.brainLayerSizes = [...layerSizes];
    this.initDefaultWorld();
    return true;
  }
}
