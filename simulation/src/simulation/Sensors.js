/**
 * Perception and Sensory Engine for Boids
 * Computes 32 sensor inputs for the neural network brain:
 * - 8 Food raycasts (360 deg)
 * - 8 Predator raycasts (360 deg)
 * - 4 Mate raycasts (360 deg)
 * - 4 Shelter raycasts (360 deg)
 * - 3 Flocking signals (Density, Alignment, Center of Mass)
 * - 5 Internal status signals (Energy, Libido, InShelter, Speed, Health)
 */
export class Sensors {
  static NUM_INPUTS = 32;

  /**
   * Calculate sensory inputs array for a given boid
   * @param {import('./Boid.js').Boid} boid 
   * @param {import('./Engine.js').Engine} engine 
   * @returns {Float32Array} 32-element input vector
   */
  static getInputs(boid, engine) {
    const inputs = new Float32Array(Sensors.NUM_INPUTS);
    let idx = 0;

    const visionRadius = boid.visionRadius;
    const heading = Math.atan2(boid.velocity.y, boid.velocity.x);

    // 1. Food Raycasts (8 directions)
    const foodRays = 8;
    const foodDistances = new Float32Array(foodRays).fill(0);
    const foodAngleStep = (Math.PI * 2) / foodRays;

    for (const food of engine.foods) {
      const dx = food.x - boid.x;
      const dy = food.y - boid.y;
      const dist = Math.hypot(dx, dy);

      if (dist <= visionRadius) {
        const angle = Math.atan2(dy, dx) - heading;
        const normAngle = Math.atan2(Math.sin(angle), Math.cos(angle)); // [-PI, PI]
        const rayIdx = Math.floor(((normAngle + Math.PI + (foodAngleStep / 2)) % (Math.PI * 2)) / foodAngleStep);
        const clampedRay = Math.max(0, Math.min(foodRays - 1, rayIdx));

        const signal = 1.0 - (dist / visionRadius); // 1 = close, 0 = far
        if (signal > foodDistances[clampedRay]) {
          foodDistances[clampedRay] = signal;
        }
      }
    }
    for (let i = 0; i < foodRays; i++) inputs[idx++] = foodDistances[i];

    // 2. Predator Raycasts (8 directions)
    const predRays = 8;
    const predDistances = new Float32Array(predRays).fill(0);
    const predAngleStep = (Math.PI * 2) / predRays;

    for (const pred of engine.predators) {
      const dx = pred.x - boid.x;
      const dy = pred.y - boid.y;
      const dist = Math.hypot(dx, dy);

      if (dist <= visionRadius * 1.2) { // Predators are visible slightly further
        const angle = Math.atan2(dy, dx) - heading;
        const normAngle = Math.atan2(Math.sin(angle), Math.cos(angle));
        const rayIdx = Math.floor(((normAngle + Math.PI + (predAngleStep / 2)) % (Math.PI * 2)) / predAngleStep);
        const clampedRay = Math.max(0, Math.min(predRays - 1, rayIdx));

        const signal = 1.0 - (dist / (visionRadius * 1.2));
        if (signal > predDistances[clampedRay]) {
          predDistances[clampedRay] = signal;
        }
      }
    }
    for (let i = 0; i < predRays; i++) inputs[idx++] = predDistances[i];

    // 3. Mate Raycasts (4 directions: Front, Right, Back, Left)
    const mateRays = 4;
    const mateDistances = new Float32Array(mateRays).fill(0);
    const mateAngleStep = (Math.PI * 2) / mateRays;

    for (const other of engine.boids) {
      if (other === boid) continue;
      // Mate criteria: opposite or compatible gender & ready to mate
      if (other.isReadyToMate() && boid.isReadyToMate() && other.gender !== boid.gender) {
        const dx = other.x - boid.x;
        const dy = other.y - boid.y;
        const dist = Math.hypot(dx, dy);

        if (dist <= visionRadius) {
          const angle = Math.atan2(dy, dx) - heading;
          const normAngle = Math.atan2(Math.sin(angle), Math.cos(angle));
          const rayIdx = Math.floor(((normAngle + Math.PI + (mateAngleStep / 2)) % (Math.PI * 2)) / mateAngleStep);
          const clampedRay = Math.max(0, Math.min(mateRays - 1, rayIdx));

          const signal = 1.0 - (dist / visionRadius);
          if (signal > mateDistances[clampedRay]) {
            mateDistances[clampedRay] = signal;
          }
        }
      }
    }
    for (let i = 0; i < mateRays; i++) inputs[idx++] = mateDistances[i];

    // 4. Shelter Raycasts (4 directions)
    const shelterRays = 4;
    const shelterDistances = new Float32Array(shelterRays).fill(0);
    const shelterAngleStep = (Math.PI * 2) / shelterRays;

    for (const shelter of engine.shelters) {
      const dx = shelter.x - boid.x;
      const dy = shelter.y - boid.y;
      const dist = Math.hypot(dx, dy);

      if (dist <= visionRadius * 1.5) {
        const angle = Math.atan2(dy, dx) - heading;
        const normAngle = Math.atan2(Math.sin(angle), Math.cos(angle));
        const rayIdx = Math.floor(((normAngle + Math.PI + (shelterAngleStep / 2)) % (Math.PI * 2)) / shelterAngleStep);
        const clampedRay = Math.max(0, Math.min(shelterRays - 1, rayIdx));

        // Signal is high when close or inside
        const signal = Math.min(1.0, 1.0 - ((dist - shelter.radius) / (visionRadius * 1.5)));
        if (signal > shelterDistances[clampedRay]) {
          shelterDistances[clampedRay] = Math.max(0, signal);
        }
      }
    }
    for (let i = 0; i < shelterRays; i++) inputs[idx++] = shelterDistances[i];

    // 5. Flocking Metrics (Density, Alignment, Center of Mass)
    let neighborCount = 0;
    let avgVx = 0;
    let avgVy = 0;
    let centerComX = 0;
    let centerComY = 0;

    for (const neighbor of engine.boids) {
      if (neighbor === boid) continue;
      const dx = neighbor.x - boid.x;
      const dy = neighbor.y - boid.y;
      const distSq = dx * dx + dy * dy;

      if (distSq <= visionRadius * visionRadius) {
        neighborCount++;
        avgVx += neighbor.velocity.x;
        avgVy += neighbor.velocity.y;
        centerComX += neighbor.x;
        centerComY += neighbor.y;
      }
    }

    if (neighborCount > 0) {
      // Density
      inputs[idx++] = Math.min(1.0, neighborCount / 15.0);

      // Alignment angle delta
      avgVx /= neighborCount;
      avgVy /= neighborCount;
      const neighborHeading = Math.atan2(avgVy, avgVx);
      const alignDelta = Math.atan2(Math.sin(neighborHeading - heading), Math.cos(neighborHeading - heading));
      inputs[idx++] = alignDelta / Math.PI; // [-1, 1]

      // Center of Mass direction
      centerComX /= neighborCount;
      centerComY /= neighborCount;
      const comAngle = Math.atan2(centerComY - boid.y, centerComX - boid.x) - heading;
      inputs[idx++] = Math.atan2(Math.sin(comAngle), Math.cos(comAngle)) / Math.PI; // [-1, 1]
    } else {
      inputs[idx++] = 0;
      inputs[idx++] = 0;
      inputs[idx++] = 0;
    }

    // 6. Internal State Signals
    inputs[idx++] = boid.energy / boid.maxEnergy; // Energy [0, 1]
    inputs[idx++] = boid.libido / boid.maxLibido; // Libido [0, 1]
    inputs[idx++] = boid.inShelter ? 1.0 : 0.0;   // In shelter flag
    const speed = Math.hypot(boid.velocity.x, boid.velocity.y);
    inputs[idx++] = Math.min(1.0, speed / boid.maxSpeed); // Current Speed
    inputs[idx++] = boid.health / boid.maxHealth; // Health [0, 1]

    return inputs;
  }
}
