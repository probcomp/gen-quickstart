<template>
<div>
  <svg :height="size.w" :width="size.w">
      <circle v-for="(point, idx) in points" :key="idx" 
          :cx="xLogicalToPixel(point.x)" 
          :cy="yLogicalToPixel(point.y)" 
          r="2"
          fill="rgba(0,0,0,0.5)" />

      <polyline v-for="(curve, j) in curvePoints" :key="j" :points="curve" :style="`fill:none;stroke:rgba(0,200,0,0.5);stroke-width:1`" />
  </svg>
</div>
</template>

<script>
export default {
  name: 'Trace',
  props: ['trace', 'info', 'size', 'tId'],

  // Computed properties can be accessed with `this.propertyName` from other
  // JavaScript, or simply as `propertyName` from within the template.
  computed: {
    // For scaling
    paddingPixels() { return 0.1 * this.size.w },
    actualSize() {return this.size.w - 2 * this.paddingPixels },
    minX() { return Math.min(...this.trace['curveXs']) },
    minY() { return Math.min(...this.trace['curveYs'][0]) },
    maxX() { return Math.max(...this.trace['curveXs']) },
    maxY() { return Math.max(...this.trace['curveYs'][0]) },
    colors() { return ["red", "blue", "green", "cyan", "orange", "purple", "lightblue", "lightgreen", "darkred", "darkgreen", "pink"]},
    
    // Packaging all info related to a point together
    points() {
      return this.info[0].map((x, i) => {
        return {x: x, y: this.trace['y-coords'][i]}
      })
    },

    curvePoints() {
      return this.trace['curveYs'].map((curve, j) => {
        return this.trace['curveXs'].map((x, i) => {
          return `${this.xLogicalToPixel(x)}, ${this.yLogicalToPixel(curve[i])}`
        }).join(' ')
      })
    }
  },

  // Methods are accessible as this.methodName(...) from other javascript, or as simply methodName(...)
  // from templates. Here, we use methods to convert between logical and pixel coordinates for x,y.
  methods: {
    xLogicalToPixel(x) {
      return this.paddingPixels + this.actualSize * ((x - this.minX) / (this.maxX - this.minX))
    },
    yLogicalToPixel(y) {
      return this.size.w - this.paddingPixels - this.actualSize * ((y - this.minY) / (this.maxY - this.minY))
    },
    xPixelToLogical(x) {
      return (x-this.paddingPixels)/this.actualSize * (this.maxX - this.minX) + this.minX
    }, 
    stdLogicalToPixel(std) {
      return std / (this.maxY - this.minY) * this.actualSize
    }
  }
}
</script>

<style scoped>
  svg {
    border: solid 1px black;
  }
</style>