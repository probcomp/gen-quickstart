<template>
  <svg :height="size.w" :width="size.w">

	<!-- Draw border -->
	<rect :x="0" :y="0" :width="size.w" :height="size.w" style="fill-opacity:0.0;stroke:black"/>

	<!-- Draw obstacles  -->
	<polygon v-for="(obs, i) in trace.scene.obstacles" :key="i"
			 :points="svgPoints(obs)" style="fill:rgba(100,100,100,1)" />

	<!-- If there is a path, draw the gray line segments that make it up -->
	<g v-if="trace.path.length > 1">
			<line v-for="n in trace.path.length-1" :key="n"
				  :x1="xLogicalToPixel(trace.path[n-1].x)" 
				  :y1="yLogicalToPixel(trace.path[n-1].y)"
				  :x2="xLogicalToPixel(trace.path[n].x)"   
				  :y2="yLogicalToPixel(trace.path[n].y)" 
				  style="stroke-width:2;stroke:rgba(100,100,100,0.4)"/>
	</g>

	<!-- Destination point -->
	<circle :cx="xLogicalToPixel(trace.dest.x)" 
			:cy="yLogicalToPixel(trace.dest.y)" 
			r="3" fill="rgba(255,0,0,0.5)" />
	
	<!-- Start point -->
	<circle :cx="xLogicalToPixel(trace.start.x)" 
			:cy="yLogicalToPixel(trace.start.y)" 
			r="3" fill="blue" />
	
	<!-- Observations -->
  <circle v-for="(m, i) in trace.measurements" 
			:key="i" 
			:cx="xLogicalToPixel(m.x)" 
			:cy="yLogicalToPixel(m.y)" 
			r="2" fill="black" />
</svg>
</template>

<script>
export default {
  name: 'Trace',
  props: ['info', 'trace', 'size'],
  methods: {
    xLogicalToPixel(x) {
      return (x - this.trace.scene.xmin) / (this.trace.scene.xmax-this.trace.scene.xmin) * this.size.w
    },
    yLogicalToPixel(y) {
      return this.size.w - ((y - this.trace.scene.ymin) / (this.trace.scene.ymax-this.trace.scene.ymin)) * this.size.w
    },
    xPixelToLogical(x) {
      return x/(this.size.w) * (this.trace.scene.xmax - this.trace.scene.xmin) + this.trace.scene.xmin
    },
    svgPoints(obs) {
      var self = this;
      return obs.vertices.map(function (pt) { return self.xLogicalToPixel(pt.x) + "," + self.yLogicalToPixel(pt.y); }).join(' ')
    }
  }
}
</script>
