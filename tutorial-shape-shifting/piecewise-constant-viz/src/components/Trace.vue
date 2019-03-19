<template>
  <!-- size.w is a suggested width; size.h is a suggested height
       Because we want to maintain a constant aspect ratio of 1:1,
       we use only size.w, and use it for both the width and height -->
  <svg :height="size.w" :width="size.w">
      <!-- data points -->
      <circle v-for="(point, idx) in points" :key="idx" 
          :cx="xLogicalToPixel(point.x)" 
          :cy="yLogicalToPixel(point.y)" 
          r="2"
          fill="rgba(0,0,0,0.5)" />

      <!-- segments -->
      <g v-for="(sgmt, idx) in segments" :key="idx">
        <line :x1="xLogicalToPixel(sgmt.x1)" :y1="yLogicalToPixel(sgmt.y)"
              :x2="xLogicalToPixel(sgmt.x2)" :y2="yLogicalToPixel(sgmt.y)"
              :style="`stroke:${colors[idx]}; stroke-width:2`" />
        <line :x1="xLogicalToPixel(sgmt.x1)" :y1="yLogicalToPixel(sgmt.y)"
              :x2="xLogicalToPixel(sgmt.x2)" :y2="yLogicalToPixel(sgmt.y)"
              :style="`stroke:rgba(0,0,0,0.1); stroke-width:${2*stdLogicalToPixel(trace.noise)}`" />
      </g>
  </svg>
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
    minX() { return Math.min(...this.info[0]) },
    minY() { return Math.min(...this.trace['y-coords']) },
    maxX() { return Math.max(...this.info[0]) },
    maxY() { return Math.max(...this.trace['y-coords']) },
    colors() { return ["red", "blue", "green", "cyan", "orange", "purple", "lightblue", "lightgreen", "darkred", "darkgreen", "pink"]},
    
    // Packaging all info related to a point together
    points() {
      return this.info[0].map((x, i) => {
        return {x: x, y: this.trace['y-coords'][i]}
      })
    },

    segments() {
      let s = []
      let total = this.minX
      for (let i = 0; i < this.trace['n']; i++) {
        let start = total
        total += this.trace['fracs'][i]*(this.maxX-this.minX)
        s.push({y: this.trace['values'][i], x1: start, x2: total})
      }
      return s
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