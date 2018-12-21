<template>
<div>
  <GenViz>
    <div slot-scope="gen">
      <h1>Traces (Grid View)</h1>
      <div id="traces">
          <div v-for="(t, tId) in gen.traces" :key="tId">
              <slot v-bind:trace="t" v-bind:info="gen.info" v-bind:size="traceSize" v-bind:tId="tId">
          </div>
      </div>
    </div>
  </GenViz>
</div>
</template>

<script>
import GenViz from './GenViz.vue'
export default {
  name: 'GridViz',
  components: {GenViz},
  data() { 
    return {
      windowSize: {height: window.innerHeight, width: window.innerWidth},
    }
  },
  computed: {
      traceSize() {
         return {h: this.windowSize.height/5, w: this.windowSize.width/5}
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

<style>
html, body {
    margin-left: 0;
    margin-right: 0;
}
</style>

<style scoped>
h1 {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    text-align: center;
}
#traces {
    display: flex;
    flex-flow: row wrap;
}

</style>