/**
 * Feedforward Neural Network for Boid AI
 * Supports up to 1000 neurons across configurable layer architectures.
 */
export class NeuralNetwork {
  /**
   * @param {number[]} layerSizes Array of neuron counts per layer, e.g. [28, 64, 32, 8]
   */
  constructor(layerSizes = [28, 64, 32, 8]) {
    this.layerSizes = [...layerSizes];
    this.totalNeurons = this.layerSizes.reduce((sum, n) => sum + n, 0);

    if (this.totalNeurons > 1000) {
      console.warn(`Neural network exceeds 1000 neurons (${this.totalNeurons}). Clamping total neurons.`);
    }

    // Weights: weights[l][i][j] is weight from neuron j in layer l to neuron i in layer l+1
    this.weights = [];
    // Biases: biases[l][i] is bias for neuron i in layer l+1
    this.biases = [];
    // Activations cache for visualization & debugging
    this.activations = [];

    this._initializeStructure();
  }

  _initializeStructure() {
    this.weights = [];
    this.biases = [];
    this.activations = [];

    for (let l = 0; l < this.layerSizes.length; l++) {
      this.activations.push(new Float32Array(this.layerSizes[l]));
      if (l < this.layerSizes.length - 1) {
        const rows = this.layerSizes[l + 1];
        const cols = this.layerSizes[l];
        const weightMatrix = [];
        const scale = Math.sqrt(2.0 / cols); // He initialization

        for (let i = 0; i < rows; i++) {
          const row = new Float32Array(cols);
          for (let j = 0; j < cols; j++) {
            row[j] = (Math.random() * 2 - 1) * scale;
          }
          weightMatrix.push(row);
        }
        this.weights.push(weightMatrix);

        const biasVector = new Float32Array(rows);
        for (let i = 0; i < rows; i++) {
          biasVector[i] = (Math.random() * 2 - 1) * 0.1;
        }
        this.biases.push(biasVector);
      }
    }
  }

  /**
   * Performs forward propagation
   * @param {number[]|Float32Array} inputs
   * @returns {Float32Array} Output activations
   */
  predict(inputs) {
    if (inputs.length !== this.layerSizes[0]) {
      throw new Error(`Input size mismatch. Expected ${this.layerSizes[0]}, got ${inputs.length}`);
    }

    // Layer 0 is inputs
    for (let i = 0; i < inputs.length; i++) {
      this.activations[0][i] = inputs[i];
    }

    // Forward pass through hidden layers to output
    for (let l = 0; l < this.weights.length; l++) {
      const currentInputs = this.activations[l];
      const nextActivations = this.activations[l + 1];
      const weightMatrix = this.weights[l];
      const biasVector = this.biases[l];
      const isOutputLayer = (l === this.weights.length - 1);

      for (let i = 0; i < nextActivations.length; i++) {
        let sum = biasVector[i];
        const row = weightMatrix[i];
        for (let j = 0; j < currentInputs.length; j++) {
          sum += row[j] * currentInputs[j];
        }

        // Activation function: Tanh for outputs, LeakyReLU for hidden
        if (isOutputLayer) {
          nextActivations[i] = Math.tanh(sum);
        } else {
          nextActivations[i] = sum > 0 ? sum : sum * 0.01; // LeakyReLU
        }
      }
    }

    return this.activations[this.activations.length - 1];
  }

  /**
   * Copies parameters from another network
   * @param {NeuralNetwork} sourceNet
   */
  copyFrom(sourceNet) {
    this.layerSizes = [...sourceNet.layerSizes];
    this.totalNeurons = sourceNet.totalNeurons;
    this._initializeStructure();

    for (let l = 0; l < sourceNet.weights.length; l++) {
      for (let i = 0; i < sourceNet.weights[l].length; i++) {
        this.weights[l][i].set(sourceNet.weights[l][i]);
      }
      this.biases[l].set(sourceNet.biases[l]);
    }
  }

  /**
   * Mutate weights and biases with a given probability and magnitude
   * @param {number} mutationRate Probability of mutating a parameter (0 to 1)
   * @param {number} mutationAmount Gaussian noise magnitude
   */
  mutate(mutationRate = 0.05, mutationAmount = 0.2) {
    for (let l = 0; l < this.weights.length; l++) {
      const weightMatrix = this.weights[l];
      const biasVector = this.biases[l];

      for (let i = 0; i < weightMatrix.length; i++) {
        const row = weightMatrix[i];
        for (let j = 0; j < row.length; j++) {
          if (Math.random() < mutationRate) {
            // Gaussian noise approximation
            const u1 = Math.random() || 0.0001;
            const u2 = Math.random() || 0.0001;
            const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            row[j] += randStdNormal * mutationAmount;
          }
        }
        if (Math.random() < mutationRate) {
          const u1 = Math.random() || 0.0001;
          const u2 = Math.random() || 0.0001;
          const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
          biasVector[i] += randStdNormal * mutationAmount;
        }
      }
    }
  }

  /**
   * Export network parameters to plain JS object
   */
  toJSON() {
    return {
      layerSizes: this.layerSizes,
      weights: this.weights.map(m => m.map(r => Array.from(r))),
      biases: this.biases.map(b => Array.from(b))
    };
  }

  /**
   * Load network parameters from object
   */
  fromJSON(data) {
    this.layerSizes = [...data.layerSizes];
    this.totalNeurons = this.layerSizes.reduce((sum, n) => sum + n, 0);
    this._initializeStructure();

    for (let l = 0; l < data.weights.length; l++) {
      for (let i = 0; i < data.weights[l].length; i++) {
        this.weights[l][i].set(data.weights[l][i]);
      }
      this.biases[l].set(data.biases[l]);
    }
  }
}
