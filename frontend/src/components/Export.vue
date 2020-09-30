<template>
  <div id="ExportResults">
    =======================================================
    <br>
    Models/Algorithms and Hyper-parameters: {{ ModelsPickled }}
    <br>
    =======================================================
  </div>
</template>

<script>

import { EventBus } from '../main.js'
import * as Cryo from 'cryo'
export default {
  name: 'Export',
  data () {
    return {
      ModelsPickled: '',
      ensModels: [],
    }
  },
  methods: {
    Pickle () {
      this.ModelsPickled = Cryo.stringify(this.ensModels)
    }
  },
  mounted () {
    EventBus.$on('ExtractResults', data => {
    this.ensModels = data})
    EventBus.$on('ExtractResults', this.Pickle)
  }
}
</script>

<style scoped>
#ExportResults {
  word-break: break-all !important;
}
</style>