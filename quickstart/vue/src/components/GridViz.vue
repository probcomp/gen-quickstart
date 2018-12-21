<template>
<div>
    <h1>Traces (Grid View)</h1>
    <div id="traces">
        <div v-for="(t, tId) in traces" :key="tId">
            <Trace :trace="t" :info="info" :size="traceSize" :tId="tId">
        </div>
    </div>
</div>
</template>

<script>
import Trace from './Trace.vue'

export default {
  name: 'GridViz',
  components: {Trace},
  data() {
    return {
      windowSize: {height: window.innerHeight, width: window.innerWidth},
    }
  },
  props: ['traces', 'info'],
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