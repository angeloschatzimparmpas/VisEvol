<template>
  <div class="text-center">
    <label id="data" for="param-dataset" data-toggle="tooltip" data-placement="right" title="Tip: use one of the data sets already provided or upload a new file.">{{ dataset }}</label>
    <select id="selectFile" @change="selectDataSet()">
        <option value="HeartC.csv" selected>Heart Disease</option>
        <option value="RiskC.csv">Risk Disease</option>
        <option value="SeismicC.csv">Seismic Disease</option>
        <option value="biodegC.csv">Biodeg Disease</option>
        <option value="ImportsC.csv">Imports Disease</option>
        <option value="local">Upload File</option>
    </select>
    <button class="btn-outline-success"
    id="initializeID"
    v-on:click="initialize">
    <font-awesome-icon icon="search" />
    {{ searchText }}
    </button>
    <button class="btn-outline-danger"
    id="resetID"
    v-on:click="reset">
    <font-awesome-icon icon="trash" />
    {{ resetText }}
    </button>
  </div>
</template>

<script>
import Papa from 'papaparse'
import { EventBus } from '../main.js'
import {$,jQuery} from 'jquery';
import * as d3Base from 'd3'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)

export default {
  name: 'DataSetExecController',
  data () {
    return {
      defaultDataSet: 'HeartC', // default value for the first data set
      searchText: 'Hyper-parameter search',
      resetText: 'Reset',
      dataset: 'Data set:'
    }
  },
  methods: {
    selectDataSet () {   
      const fileName = document.getElementById('selectFile')
      this.defaultDataSet = fileName.options[fileName.selectedIndex].value
      this.defaultDataSet = this.defaultDataSet.split('.')[0]

      if (this.defaultDataSet == "HeartC" || this.defaultDataSet == "RiskC" || this.defaultDataSet == "SeismicC" || this.defaultDataSet == "biodegC" || this.defaultDataSet == "ImportsC") { // This is a function that handles a new file, which users can upload.
        this.dataset = "Data set"
        d3.select("#data").select("input").remove(); // Remove the selection field.
        EventBus.$emit('SendToServerDataSetConfirmation', this.defaultDataSet)
      } else {
        EventBus.$emit('SendToServerDataSetConfirmation', this.defaultDataSet)
        d3.select("#data").select("input").remove();
        this.dataset = ""
        var data
        d3.select("#data")
          .append("input")
          .attr("type", "file")
          .style("font-size", "18.5px")
          .style("width", "200px")
          .on("change", function() {
            var file = d3.event.target.files[0];
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                  data = results.data;
                  EventBus.$emit('SendToServerLocalFile', data)
                }
              });
          })
      }
    },
    reset () {
      EventBus.$emit('reset')
      EventBus.$emit('alternateFlagLock')
    },
    initialize () {
      EventBus.$emit('ConfirmDataSet')
    }
  }
}
</script>
