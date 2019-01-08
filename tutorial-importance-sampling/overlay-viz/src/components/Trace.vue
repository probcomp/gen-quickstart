<template>
  <svg :height="size.w" :width="size.w">

  <!-- GLOBAL ELEMENTS -->

	<!-- Border -->
	<rect :x="0" :y="0" :width="size.w" :height="size.w" style="fill-opacity:0.0;stroke:black"/>

	<!-- Obstacles -->
  <g v-if="containsKey(info, 'scene')" >
    <polygon v-for="(obs, i) in info.scene.obstacles" :key="i"
        :points="svgPoints(obs)" style="fill:rgba(100,100,100,1)" />
  </g>

	<!-- Start point -->
  <g v-if="containsKey(info, 'start')">
    <circle :cx="xLogicalToPixel(info.start.x)" 
        :cy="yLogicalToPixel(info.start.y)" 
        r="10" fill="blue" />
  </g>>

	<!-- Destination point -->
 <g v-if="containsKey(info, 'dest')">
    <circle :cx="xLogicalToPixel(info.dest.x)" 
        :cy="yLogicalToPixel(info.dest.y)" 
        r="10" fill="red" />
  </g>
	
	<!-- Measurements -->
  <g v-if="containsKey(info, 'measurements')">
    <circle v-for="(m, i) in info.measurements"
        :key="i" 
        :cx="xLogicalToPixel(m.x)" 
        :cy="yLogicalToPixel(m.y)" 
        r="5" fill="black" />
  </g>

  <!-- RRT tree -->
  <g v-if="containsKey(info, 'tree_edges')">
    <line v-for="(edge, i) in info.tree_edges" :key="i"
          :x1="xLogicalToPixel(edge[0].x)" :y1="yLogicalToPixel(edge[0].y)"
          :x2="xLogicalToPixel(edge[1].x)" :y2="yLogicalToPixel(edge[1].y)"
          style="stroke:rgb(0,0,0);stroke-width:2" 
    />
  </g>

  <!-- Path -->
  <g v-if="containsKey(info, 'path_edges')">
    <line v-for="(edge, i) in info.path_edges" :key="i"
          :x1="xLogicalToPixel(edge[0].x)" :y1="yLogicalToPixel(edge[0].y)"
          :x2="xLogicalToPixel(edge[1].x)" :y2="yLogicalToPixel(edge[1].y)"
          style="stroke:rgb(255,140,0);stroke-width:6" 
    />
  </g>

  <!-- Tiles -->
  <g v-if="containsKey(info, 'tiles')">
    <rect v-for="(tile, i) in info.tiles"
      :key="i"
      :x="xLogicalToPixel(tile.x)"
      :y="yLogicalToPixel(tile.y + tile.h)"
      :width ="tile.w * size.w / (info.scene.xmax - info.scene.xmin)"
      :height="tile.h * size.w / (info.scene.ymax - info.scene.ymin)"
      :style="{fillOpacity: tile.density * 0.7, fill: 'red'}" />
  </g>

  <!-- PER TRACE ELEMENTS -->
	<g v-for="(trace, i) in traces" :key="i">

		<!-- Destination point -->
		<circle :cx="xLogicalToPixel(trace.dest.x)" 
				:cy="yLogicalToPixel(trace.dest.y)" 
				r="5" fill="rgba(255,0,0,0.5)" />
	</g>



</svg>
</template>

<script>
export default {
  name: 'Trace',
  props: ['info', 'traces'],
  methods: {
    xLogicalToPixel(x) {
      return (x - this.info.scene.xmin) / (this.info.scene.xmax-this.info.scene.xmin) * this.size.w
    },
    yLogicalToPixel(y) {
      return this.size.w - ((y - this.info.scene.ymin) / (this.info.scene.ymax-this.info.scene.ymin)) * this.size.w
    },
    xPixelToLogical(x) {
      return x/(this.size.w) * (this.info.scene.xmax - this.info.scene.xmin) + this.info.scene.xmin
    },
    svgPoints(obs) {
      var self = this;
      return obs.vertices.map(function (pt) { return self.xLogicalToPixel(pt.x) + "," + self.yLogicalToPixel(pt.y); }).join(' ')
    },
    containsKey(obj, key) {
      return Object.keys(obj).includes(key);
    }
	},
	data() {
		return {
			windowSize: {height: window.innerHeight, width: window.innerWidth},
		}
	},
  computed: {
    size() {
      return {h: this.windowSize.height/2, w: this.windowSize.width/2}
    }
	 },
	mounted () {
     this.$nextTick(() => {
         window.addEventListener('resize', () => {
             this.windowSize = {height: window.innerHeight, width: window.innerWidth};
         })
     })
   }

}
</script>
