export class Food {
  static idCounter = 0;

  constructor(x, y, energyValue = 35) {
    this.id = ++Food.idCounter;
    this.x = x;
    this.y = y;
    this.radius = 4;
    this.energyValue = energyValue;
    this.pulseAngle = Math.random() * Math.PI * 2;
  }

  update() {
    this.pulseAngle += 0.05;
  }
}
