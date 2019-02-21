<template>
  <svg :height="size.w" :width="size.w">
      <circle
          v-for="n in info.num"
          :key="n"
          :cx="xLogicalToPixel(info.xs[n-1])"
          :cy="yLogicalToPixel(info.ys[n-1])" r="3"
      />

      <line
          :x1="0"
          :y1="yLogicalToPixel(xPixelToLogical(0)*trace.slope + trace.intercept)"
          :x2="size.w"
          :y2="yLogicalToPixel(xPixelToLogical(size.w) * trace.slope + trace.intercept)"
          style="stroke:rgba(0,0,0,0.7);stroke-width:2"
      />
  </svg>
</template>

<script>
export default {
  name: 'Trace',
  props: ['info', 'trace', 'size', 'tId'],
  methods: {
    xLogicalToPixel(x) {
      return (x - this.info.xlim[0]) / (this.info.xlim[1]-this.info.xlim[0]) * this.size.w
    },
    yLogicalToPixel(y) {
      return this.size.w - ((y - this.info.ylim[0]) / (this.info.ylim[1]-this.info.ylim[0])) * this.size.w
    },
    xPixelToLogical(x) {
      return x/(this.size.w) * (this.info.xlim[1] - this.info.xlim[0]) + this.info.xlim[0]
    }
  }
}
</script>
