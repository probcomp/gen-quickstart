<template>
  <div ref="genviz">
      <GridViz :traces="traces" :info="info" />
  </div>
</template>

<script>
function uuid4() {
  return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
      (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  )
}

window.clientId = uuid4()
window.vizId = window.location.pathname.replace(/\//g, '')
window.socket = new WebSocket("ws://" + window.location.hostname + ":" + window.location.port)
window.onbeforeunload = function(){
    window.socket.send(JSON.stringify({"clientId": window.clientId, "vizId": window.vizId, "action": "disconnect"}))
    window.socket.close();
}

window.getCSS = function() {
  var css= [];
  for (var sheeti= 0; sheeti<document.styleSheets.length; sheeti++) {
      var sheet= document.styleSheets[sheeti];
      var rules= ('cssRules' in sheet)? sheet.cssRules : sheet.rules;
      for (var rulei= 0; rulei<rules.length; rulei++) {
          var rule= rules[rulei];
          if ('cssText' in rule)
              css.push(rule.cssText);
          else
              css.push(rule.selectorText+' {\n'+rule.style.cssText+'\n}\n');
      }
  }
  return css.join('\n');
}

import GridViz from './GridViz.vue'

export default {
  name: 'GenViz',
  components: {GridViz},
  data() { 
    return {
      traces: {},
      info: {}
    }
  },
  methods: {
    initialize(info, traces) {
      this.info = info;
      this.traces = traces;
    },
    putTrace(tId, t) {
      this.$set(this.traces, tId, t)
    },
    removeTrace(tId) {
      this.$delete(this.traces, tId)
    },
    sendHTML() {
      let html = this.$refs["genviz"].innerHTML + `<style>${window.getCSS()}</style>`
      window.socket.send(JSON.stringify({"clientId": window.clientId, "vizId": window.vizId, "action": "save", "content": html}))
    }
  },
  mounted () {
    window.socket.onopen = () => {
      window.socket.onmessage = ({data}) => {
        let msg = JSON.parse(data);
        switch (msg.action) {
          case 'initialize':
            this.initialize(msg.info, msg.traces)
            break;
          case 'putTrace':
            this.putTrace(msg.tId, msg.t)
            break;
          case 'removeTrace':
            this.removeTrace(msg.tId)
            break;
          case 'saveHTML':
            this.sendHTML()
            break;
        }
      }
      window.socket.send(JSON.stringify({"clientId": window.clientId, "vizId": window.vizId, "action": "connect"}))
    }
  }
}
</script>