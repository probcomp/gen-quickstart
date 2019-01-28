<template>
  <!-- size.w is a suggested width; size.h is a suggested height
       Because we want to maintain a constant aspect ratio of 1:1,
       we use only size.w, and use it for both the width and height -->
  <svg :height="size.w" :width="size.w" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" >
      <!-- data points -->
      <circle v-for="(point, idx) in points" :key="idx" 
          :cx="xLogicalToPixel(point.x)" 
          :cy="yLogicalToPixel(point.y)" 
          r="3"
          :fill="point.is_outlier ? 'red' : 'blue'" />

      <!-- inlier noise -->
      <polygon
        :points="inlierNoisePolygonPoints" 
        style="fill:rgba(0,0,0,0.3)" />

      <!-- outlier noise -->
      <line 
        :x1="0"       :y1="yLogicalToPixel(0.)" 
        :x2="size.w"  :y2="yLogicalToPixel(0.)" 
        :style="'stroke:rgba(0,0,0,0.1);stroke-width:' + stdLogicalToPixel(10.0)*4" />

      <!-- mean -->
      <line :x1="-200" :y1="yLogicalToPixel(xPixelToLogical(-200)*trace['slope'] + trace['intercept'])" 
            :x2="size.w+200"  :y2="yLogicalToPixel(xPixelToLogical(size.w+200) *trace['slope'] + trace['intercept'])" 
              style="stroke:rgba(0,0,0,0.7);stroke-width:2" />
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

    // Packaging all info related to a point together
    points() {
      return this.info[0].map((x, i) => {
        return {x: x, y: this.trace['y-coords'][i], is_outlier: this.trace['outliers'][i]}
      })
    },

    // Calculate vertices of the polynomial we use to show inlier noise
    inlierNoisePolygonPoints() {
      let x1_pixel = -200
      let y1_no_noise = this.xPixelToLogical(x1_pixel) * this.trace['slope'] + this.trace['intercept']
      let y1_high_pixel = this.yLogicalToPixel(y1_no_noise + this.trace['inlier_std'] * 2)
      let y1_low_pixel  = this.yLogicalToPixel(y1_no_noise - this.trace['inlier_std'] * 2)

      let x2_pixel = this.size.w + 200
      let y2_no_noise = this.xPixelToLogical(x2_pixel) * this.trace['slope'] + this.trace['intercept']
      let y2_high_pixel = this.yLogicalToPixel(y2_no_noise + this.trace['inlier_std'] * 2)
      let y2_low_pixel  = this.yLogicalToPixel(y2_no_noise - this.trace['inlier_std'] * 2)
      
      return `${x1_pixel},${y1_high_pixel} ${x2_pixel},${y2_high_pixel} ${x2_pixel},${y2_low_pixel} ${x1_pixel},${y1_low_pixel}`
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
