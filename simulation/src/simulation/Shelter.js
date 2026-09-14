export class Shelter {
  static idCounter = 0;

  constructor(x, y, radius = 70) {
    this.id = ++Shelter.idCounter;
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.pulse = 0;
  }

  containsPoint(px, py) {
    const dx = px - this.x;
    const dy = py - this.y;
    return (dx * dx + dy * dy) <= (this.radius * this.radius);
  }

  update() {
    this.pulse = (this.pulse + 0.02) % (Math.PI * 2);
  }
}
