<template>
<div>
  <div align="center">
            Projection method: <select id="selectBarChart" @change="selectVisualRepresentation()">
              <option value="umap" selected>UMAP</option>
              <option value="mds">MDS</option>
              <option value="tsne">t-SNE</option>
            </select>
            &nbsp;&nbsp;
            Action: 
            <button
            id="Remove" style="visibility: visible"
            v-on:click="Remove">
            <font-awesome-icon icon="dna" />
            {{ CrossoverMutateText }}
            </button>
            <button style="display: none"
            id="AddEnsemble"
            v-on:click="Add">
            <font-awesome-icon icon="plus-square" />
            {{ AddEnsemble }}
            </button>

  </div>
  <div id="OverviewPlotly" class="OverviewPlotly"></div>
</div>
</template>

<script>
import * as Plotly from 'plotly.js'
import * as d3Base from 'd3'

import { EventBus } from '../main.js'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)

export default {
  name: 'HyperParameterSpace',
  data () {
    return {
      CrossoverMutateText: 'Crossover and mutate unselected models',
      AddEnsemble: 'Add selected models to ensemble',
      WH: [],
      ScatterPlotResults: '',
      representationDef: 'umapCM',
      FlagFinalStage: 0,
      //RemainingPointsIndices: []
    }
  },
  methods: {
    reset () {
      this.ScatterPlotResults = ''
      Plotly.purge('OverviewPlotly')
    },
    clean(obj) {
      var propNames = Object.getOwnPropertyNames(obj);
      for (var i = 0; i < propNames.length; i++) {
        var propName = propNames[i];
        if (obj[propName] === null || obj[propName] === undefined) {
          delete obj[propName];
        }
      }
    },
    selectVisualRepresentation () {
      const representationSelectionDocum = document.getElementById('selectBarChart')
      this.representationSelection = representationSelectionDocum.options[representationSelectionDocum.selectedIndex].value
      EventBus.$emit('RepresentationSelection', this.representationSelection)
    },
    ScatterPlotView () {
      Plotly.purge('OverviewPlotly')

      //if (this.FlagFinalStage == 0) {
      var modelId = JSON.parse(this.ScatterPlotResults[0])
      var colorsforScatterPlot = JSON.parse(this.ScatterPlotResults[1])
      var parametersLoc = JSON.parse(this.ScatterPlotResults[2])
      var parameters = JSON.parse(parametersLoc)
      var MDSData= JSON.parse(this.ScatterPlotResults[9])
      var TSNEData = JSON.parse(this.ScatterPlotResults[10])
      var UMAPData = JSON.parse(this.ScatterPlotResults[11])
      // } else {
      //   console.log('mpike')
      //   var modelId = this.RemainingPointsIndices.map((item) => JSON.parse(this.ScatterPlotResults[0])[item])
      //   console.log(modelId)
      //   var colorsforScatterPlot = this.RemainingPointsIndices.map((item) => JSON.parse(this.ScatterPlotResults[1])[item])
      //   var parametersLoc = this.RemainingPointsIndices.map((item) => JSON.parse(this.ScatterPlotResults[2])[item])
      //   var parameters = JSON.parse(parametersLoc)
      //   var MDSData = this.RemainingPointsIndices.map((item) => JSON.parse(this.ScatterPlotResults[9])[item])
      //   var TSNEData = this.RemainingPointsIndices.map((item) => JSON.parse(this.ScatterPlotResults[10])[item])
      //   var UMAPData = this.RemainingPointsIndices.map((item) => JSON.parse(this.ScatterPlotResults[11])[item])
      //   console.log(modelId)
      // }
      
      EventBus.$emit('sendPointsNumber', modelId.length)

      var stringParameters = []
      for (let i = 0; i < parameters.length; i++) {
        this.clean(parameters[i])
        stringParameters.push(JSON.stringify(parameters[i]).replace(/,/gi, '<br>'))
      }

      var classifiersInfoProcessing = []
      for (let i = 0; i < modelId.length; i++) {
        let tempSplit = modelId[i].split(/([0-9]+)/)
        if (tempSplit[0] == 'KNN' || tempSplit[0] == 'KNNC' || tempSplit[0] == 'KNNM' || tempSplit[0] == 'KNNCC' || tempSplit[0] == 'KNNCM' || tempSplit[0] == 'KNNMC' || tempSplit[0] == 'KNNMM') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> k-nearest neighbor' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else if (tempSplit[0] == 'LR' || tempSplit[0] == 'LRC' || tempSplit[0] == 'LRM' || tempSplit[0] == 'LRCC' || tempSplit[0] == 'LRCM' || tempSplit[0] == 'LRMC' || tempSplit[0] == 'LRMM') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> logistic regression' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else if (tempSplit[0] == 'MLP' || tempSplit[0] == 'MLPC' || tempSplit[0] == 'MLPM' || tempSplit[0] == 'MLPCC' || tempSplit[0] == 'MLPCM' || tempSplit[0] == 'MLPMC' || tempSplit[0] == 'MLPMM') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> multilayer perceptron' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else if (tempSplit[0] == 'RF' || tempSplit[0] == 'RFC' || tempSplit[0] == 'RFM' || tempSplit[0] == 'RFCC' || tempSplit[0] == 'RFCM' || tempSplit[0] == 'RFMC' || tempSplit[0] == 'RFMM') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> random forest' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> gradient boosting' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
      }

      var DataGeneral, maxX, minX, maxY, minY, layout

      var width = this.WH[0]*6.5 // interactive visualization
      var height = this.WH[1]*0.9 // interactive visualization

      if (this.representationDef == 'mds') {
        maxX = Math.max(MDSData[0])
        minX = Math.min(MDSData[0])
        maxY = Math.max(MDSData[1])
        minY = Math.max(MDSData[1])

        DataGeneral = [{
          type: 'scatter',
          mode: 'markers',
          x: MDSData[0],
          y: MDSData[1],
          hovertemplate: 
                "%{text}<br><br>" +
                "<extra></extra>",
          text: classifiersInfoProcessing,
          marker: {
           line: { color: 'rgb(0, 0, 0)', width: 3 },
            color: colorsforScatterPlot,
            size: 12,
            colorscale: 'Viridis',
            colorbar: {
              title: '# Ov. Performance (%) #',
              titleside:'right',
            },
          }
        
        }]
        layout = {

          xaxis: {
              visible: false,
              range: [minX, maxX]
          },
          yaxis: {
              visible: false,
              range: [minY, maxY]
          },
          font: { family: 'Helvetica', size: 22, color: '#000000' },
          autosize: true,
          width: width,
          height: height,
          dragmode: 'lasso',
          hovermode: "closest",
          hoverlabel: { bgcolor: "#FFF" },
          legend: {orientation: 'h', y: -0.3},
          margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 30,
            pad: 0
          },
        }
      } else if (this.representationDef == 'tsne') {
        var result = TSNEData.reduce(function(r, a) {
            a.forEach(function(s, i) {
                var key = i === 0 ? 'Xax' : 'Yax';

                r[key] || (r[key] = []); // if key not found on result object, add the key with empty array as the value

                r[key].push(s);
            })
            return r;
        }, {})
        
        maxX = Math.max(result.Xax)
        minX = Math.min(result.Xax)
        maxY = Math.max(result.Yax)
        minY = Math.max(result.Yax)

        DataGeneral = [{
          type: 'scatter',
          mode: 'markers',
          x: result.Xax,
          y: result.Yax,
          hovertemplate: 
                "%{text}<br><br>" +
                "<extra></extra>",
          text: classifiersInfoProcessing,
          marker: {
              line: { color: 'rgb(0, 0, 0)', width: 3 },
              color: colorsforScatterPlot,
              size: 12,
              colorscale: 'Viridis',
              colorbar: {
                title: '# Ov. Performance (%) #',
                titleside: 'right'
              },
          }
        }]
        layout = {

          xaxis: {
              visible: false,
              range: [minX, maxX]
          },
          yaxis: {
              visible: false,
              range: [minY, maxY]
          },
          font: { family: 'Helvetica', size: 22, color: '#000000' },
          autosize: true,
          width: width,
          height: height,
          dragmode: 'lasso',
          hovermode: "closest",
          hoverlabel: { bgcolor: "#FFF" },
          legend: {orientation: 'h', y: -0.3},
          margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 30,
            pad: 0
          },
        }

      } else {
        maxX = Math.max(UMAPData[0])
        minX = Math.min(UMAPData[0])
        maxY = Math.max(UMAPData[1])
        minY = Math.max(UMAPData[1])

        DataGeneral = [{
          type: 'scatter',
          mode: 'markers',
          x: UMAPData[0],
          y: UMAPData[1],
          hovertemplate: 
                "%{text}<br><br>" +
                "<extra></extra>",
          text: classifiersInfoProcessing,
          marker: {
           line: { color: 'rgb(0, 0, 0)', width: 3 },
            color: colorsforScatterPlot,
            size: 12,
            colorscale: 'Viridis',
            colorbar: {
              title: '# Ov. Performance (%) #',
              titleside: 'right'
            },
          }
        
        }]
        layout = {

          xaxis: {
              visible: false,
              range: [minX, maxX]
          },
          yaxis: {
              visible: false,
              range: [minY, maxY]
          },
          font: { family: 'Helvetica', size: 22, color: '#000000' },
          autosize: true,
          width: width,
          height: height,
          dragmode: 'lasso',
          hovermode: "closest",
          hoverlabel: { bgcolor: "#FFF" },
          legend: {orientation: 'h', y: -0.3},
          margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 30,
            pad: 0
          },
        }
      }
     
      var config = {scrollZoom: true, displaylogo: false, showLink: false, showSendToCloud: false, modeBarButtonsToRemove: ['toImage', 'toggleSpikelines', 'autoScale2d', 'hoverClosestGl2d','hoverCompareCartesian','select2d','hoverClosestCartesian','zoomIn2d','zoomOut2d','zoom2d'], responsive: true}
      
      var scat = document.getElementById('OverviewPlotly')
      
      Plotly.newPlot(scat, DataGeneral, layout, config)
      this.selectedPointsOverview()
    },
    selectedPointsOverview () {
      const OverviewPlotly = document.getElementById('OverviewPlotly')
      var allModels = JSON.parse(this.ScatterPlotResults[0])
      var ResultsAll = this.ScatterPlotResults
      OverviewPlotly.on('plotly_selected', function (evt) {
        if (typeof evt !== 'undefined') {
          var pushModelsRemainingTemp = []
          var indices = []
          const ClassifierIDsList = []
          for (let i = 0; evt.points.length; i++) {
            if (evt.points[i] === undefined) {
              break
            } else {
              const OnlyId = evt.points[i].text.split(' ')[2]
              const OnlyIdCleared = OnlyId.split('<br>')
              ClassifierIDsList.push(OnlyIdCleared[0])
            }
          }
          for (let i = 0; i < allModels.length; i++) {
            if (ClassifierIDsList.indexOf((allModels[i])) < 0) {
              pushModelsRemainingTemp.push(allModels[i])
              //indices.push(i)
            }
          }
          //EventBus.$on('RemainingPointsIndices', indices)
          EventBus.$emit('RemainingPoints', pushModelsRemainingTemp)
          EventBus.$emit('callValidationData', ResultsAll)
          EventBus.$emit('SendSelectedPointsUpdateIndicator', ClassifierIDsList)
        }
      })
    },
    Remove () {
      EventBus.$emit('InitializeCrossoverMutation')
    },
    Add () {
      EventBus.$emit('ChangeKey', 3)
      EventBus.$emit('InitializeCrossoverMutation')
    },
    swapButtons () {
      var add = document.getElementById('AddEnsemble');
      var remove = document.getElementById('Remove');
      remove.style.display = 'none'
      add.style.display = 'inline'
    }
  },
  mounted() {
    EventBus.$on('RemainingPointsIndices', data => { this.RemainingPointsIndices = data })
    EventBus.$on('ChangeKey', data => { this.FlagFinalStage = data })

    EventBus.$on('emittedEventCallingScatterPlot', data => {
      this.ScatterPlotResults = data})
    EventBus.$on('emittedEventCallingScatterPlot', this.ScatterPlotView)

    EventBus.$on('RepresentationSelection', data => {this.representationDef = data})
    EventBus.$on('RepresentationSelection', this.ScatterPlotView)

    EventBus.$on('Responsive', data => {
    this.WH = data})
    EventBus.$on('ResponsiveandChange', data => {
    this.WH = data})

    EventBus.$on('hideCrossMut', this.swapButtons)

    // reset view
    EventBus.$on('resetViews', this.reset)
  }
}
</script>

<style>
.infolayer {
  transform: scale(0.7) translate(520px, 110px) !important;
}
</style>

