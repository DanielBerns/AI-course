import { NeuralNetwork } from './NeuralNetwork.js';

/**
 * Handles crossover and genetic mutation of Boid neural networks.
 */
export class Genome {
  /**
   * Performs uniform crossover between two parent networks to create a child network.
   * @param {NeuralNetwork} parentA 
   * @param {NeuralNetwork} parentB 
   * @param {number} mutationRate 
   * @param {number} mutationAmount 
   * @returns {NeuralNetwork}
   */
  static crossover(parentA, parentB, mutationRate = 0.05, mutationAmount = 0.2) {
    const childNet = new NeuralNetwork(parentA.layerSizes);

    for (let l = 0; l < parentA.weights.length; l++) {
      const wA = parentA.weights[l];
      const wB = parentB.weights[l];
      const wChild = childNet.weights[l];

      const bA = parentA.biases[l];
      const bB = parentB.biases[l];
      const bChild = childNet.biases[l];

      for (let i = 0; i < wA.length; i++) {
        for (let j = 0; j < wA[i].length; j++) {
          // Uniform crossover: randomly pick gene from parent A or B
          wChild[i][j] = Math.random() < 0.5 ? wA[i][j] : wB[i][j];
        }
        bChild[i] = Math.random() < 0.5 ? bA[i] : bB[i];
      }
    }

    // Apply mutation to child
    childNet.mutate(mutationRate, mutationAmount);
    return childNet;
  }

  /**
   * Creates a mutated clone of a single parent network (asexual reproduction or starter population)
   * @param {NeuralNetwork} parent 
   * @param {number} mutationRate 
   * @param {number} mutationAmount 
   * @returns {NeuralNetwork}
   */
  static cloneAndMutate(parent, mutationRate = 0.05, mutationAmount = 0.2) {
    const childNet = new NeuralNetwork(parent.layerSizes);
    childNet.copyFrom(parent);
    childNet.mutate(mutationRate, mutationAmount);
    return childNet;
  }
}
