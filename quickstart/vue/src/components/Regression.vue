<template>
  <GridViz>
    <div slot-scope="viz">
        <svg :height="viz.size.w" :width="viz.size.w">

            <!-- data points -->
            <circle
                v-for="n in viz.info.num"
                :key="n"
                :cx="xLogicalToPixel(viz.info.xs[n], viz.info.xlim, viz.size)" 
                :cy="yLogicalToPixel(viz.info.ys[n], viz.info.ylim, viz.size)" r="3"
            />

            <!-- line -->
            <line
                :x1="0"
                :y1="yLogicalToPixel(xPixelToLogical(0, viz.info.xlim, viz.size)*viz.trace['slope'] + viz.trace['intercept'], viz.info.ylim, viz.size)"
                :x2="viz.size.w"
                :y2="yLogicalToPixel(xPixelToLogical(viz.size.w, viz.info.xlim, viz.size) *viz.trace['slope'] + viz.trace['intercept'], viz.info.ylim, viz.size)"
                style="stroke:rgba(0,0,0,0.7);stroke-width:2"
            />
        </svg>
    </div>
  </GridViz>
</template>

<script>
import GridViz from './GridViz.vue'

export default {
  name: 'Regression',
  components: {GridViz},
  methods: {
    xLogicalToPixel(x, xlim, sz) {
      return (x - xlim[0]) * sz.w
    },
    yLogicalToPixel(y, ylim, sz) {
      return sz.w - ((y - ylim[0]) * sz.w)
    },
    xPixelToLogical(x, xlim, sz) {
      return (x - 0)/(sz.w) + xlim[0]
    }
  }
}
</script>
