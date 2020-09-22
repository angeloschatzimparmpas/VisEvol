<template>
<div>
  <div align="center">
            Projection method: <select id="selectBarChartCM" @change="selectVisualRepresentationCM()">
              <option value="mdsCM" selected>MDS</option>
              <option value="tsneCM">t-SNE</option>
              <option value="umapCM">UMAP</option>
            </select>
            &nbsp;&nbsp;
            Actions: <button
            id="RemoveEnsembleID"
            v-on:click="RemoveEnsemble">
            <font-awesome-icon icon="eraser" />
            {{ CrossoverMutateText }}
            </button>
            &nbsp;&nbsp;
            <button 
            id="updateActiveScatter"
            v-on:click="sendUpdateActiveScatter">
            <font-awesome-icon icon="calculator" />
            {{ valueActive }}
            </button>
  </div>
  <div id="OverviewPlotlyCM" class="OverviewPlotlyCM"></div>
</div>
</template>

<script>
import * as Plotly from 'plotly.js'
import * as d3Base from 'd3'

import { EventBus } from '../main.js'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)

export default {
  name: 'Ensemble',
  data () {
    return {
      CrossoverMutateText: 'Remove unselected models from ensemble',
      WH: [],
      ScatterPlotResults: '',
      representationDef: 'mdsCM',
      storeEnsembleLoc: [],
      valueActive: 'Compute performance for active ensemble',
      pushModelsTempCMSame: [],
    }
  },
  methods: {
    reset () {
      this.ScatterPlotResults = ''
      this.storeEnsembleLoc = []
      Plotly.purge('OverviewPlotlyCM')
    },
    selectVisualRepresentationCM () {
      const representationSelectionDocum = document.getElementById('selectBarChartCM')
      this.representationSelection = representationSelectionDocum.options[representationSelectionDocum.selectedIndex].value
      EventBus.$emit('RepresentationSelectionCM', this.representationSelection)
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
    ScatterPlotView () {
      Plotly.purge('OverviewPlotlyCM')

      var modelId = JSON.parse(this.ScatterPlotResults[0])
      var colorsforScatterPlot = JSON.parse(this.ScatterPlotResults[1])
      var parametersLoc = JSON.parse(this.ScatterPlotResults[2])
      var parameters = JSON.parse(parametersLoc)
      var MDSData= JSON.parse(this.ScatterPlotResults[9])
      var TSNEData = JSON.parse(this.ScatterPlotResults[10])
      var UMAPData = JSON.parse(this.ScatterPlotResults[11])
      var mergedStoreEnsembleLoc = [].concat.apply([], this.storeEnsembleLoc)
      var mergedStoreEnsembleLocFormatted = []
      for (let i = 0; i < mergedStoreEnsembleLoc.length; i++) {
        mergedStoreEnsembleLocFormatted.push(parseInt(mergedStoreEnsembleLoc[i].replace(/\D/g,'')))
      }

      modelId = mergedStoreEnsembleLocFormatted.map((item) => modelId[item])
      colorsforScatterPlot = mergedStoreEnsembleLocFormatted.map((item) => colorsforScatterPlot[item])
      parameters = mergedStoreEnsembleLocFormatted.map((item) => parameters[item])
      MDSData[0] = mergedStoreEnsembleLocFormatted.map((item) => MDSData[0][item])
      MDSData[1] = mergedStoreEnsembleLocFormatted.map((item) => MDSData[1][item])
      TSNEData = mergedStoreEnsembleLocFormatted.map((item) => TSNEData[item])
      UMAPData[0] = mergedStoreEnsembleLocFormatted.map((item) => UMAPData[0][item])
      UMAPData[1] = mergedStoreEnsembleLocFormatted.map((item) => UMAPData[1][item])

      EventBus.$emit('sendPointsNumberCM', modelId.length)

      var stringParameters = []
      for (let i = 0; i < parameters.length; i++) {
        this.clean(parameters[i])
        stringParameters.push(JSON.stringify(parameters[i]).replace(/,/gi, '<br>'))
      }

      var classifiersInfoProcessing = []
      for (let i = 0; i < modelId.length; i++) {
        let tempSplit = modelId[i].split(/([0-9]+)/)
        if (tempSplit[0] == 'KNN' || tempSplit[0] == 'KNN_C' || tempSplit[0] == 'KNN_M') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> k-nearest neighbor' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else if (tempSplit[0] == 'LR' || tempSplit[0] == 'LR_C' || tempSplit[0] == 'LR_M') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> logistic regression' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else if (tempSplit[0] == 'MLP' || tempSplit[0] == 'MLP_C' || tempSplit[0] == 'MLP_M') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> multilayer perceptron' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else if (tempSplit[0] == 'RF' || tempSplit[0] == 'RF_C' || tempSplit[0] == 'RF_M') {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> random forest' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
        else {
          classifiersInfoProcessing[i] = '<b>Model ID:</b> ' + modelId[i] + '<br><b>Algorithm:</b> gradient boosting' + '<br><b>Parameters:</b> ' + stringParameters[i]
        }
      }

      var DataGeneral, maxX, minX, maxY, minY, layout

      var width = this.WH[0]*6.5 // interactive visualization
      var height = this.WH[1]*0.9 // interactive visualization

      if (this.representationDef == 'mdsCM') {
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
            size: 16,
            colorscale: 'Viridis',
            colorbar: {
              title: '# Performance (%) #',
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
          font: { family: 'Helvetica', size: 16, color: '#000000' },
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
      } else if (this.representationDef == 'tsneCM') {
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
              size: 16,
              colorscale: 'Viridis',
              colorbar: {
                title: '# Performance (%) #',
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
            size: 16,
            colorscale: 'Viridis',
            colorbar: {
              title: '# Performance (%) #',
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
      
      var scat = document.getElementById('OverviewPlotlyCM')
      
      Plotly.newPlot(scat, DataGeneral, layout, config)
      this.selectedPointsOverview()
    },
    selectedPointsOverview () {
        const OverviewPlotly = document.getElementById('OverviewPlotlyCM')
        var allModels = JSON.parse(this.ScatterPlotResults[0])
        OverviewPlotly.on('plotly_selected', function (evt) {
          if (typeof evt !== 'undefined') {
            var pushModelsRemainingTempCM = []
            const ClassifierIDsListCM = []
            for (let i = 0; evt.points.length; i++) {
              if (evt.points[i] === undefined) {
                break
              } else {
                const OnlyId = evt.points[i].text.split(' ')[2]
                const OnlyIdCleared = OnlyId.split('<br>')
                ClassifierIDsListCM.push(OnlyIdCleared[0])
              }
            }
            for (let i = 0; i < allModels.length; i++) {
              if (ClassifierIDsListCM.indexOf((allModels[i])) < 0) {
                pushModelsRemainingTempCM.push(allModels[i])
              }
            }
              EventBus.$emit('RemainingPointsCM', pushModelsRemainingTempCM)
              EventBus.$emit('SendSelectedPointsUpdateIndicatorCM', ClassifierIDsListCM)
              EventBus.$emit('SendSelectedPointsToServerEventCM', ClassifierIDsListCM)
          }
        })
      },
      RemoveEnsemble () {
        EventBus.$emit('RemoveFromEnsemble', this.pushModelsTempCMSame)
      },
      sendUpdateActiveScatter () {
        EventBus.$emit('sendToServerSelectedScatter')
      }
  },
  mounted() {

    EventBus.$on('SendSelectedPointsUpdateIndicatorCM', data => { this.pushModelsTempCMSame = data })

    EventBus.$on('SendStoredEnsemble', data => { this.storeEnsembleLoc = data })
    
    EventBus.$on('emittedEventCallingCrossoverMutation', data => {
      this.ScatterPlotResults = data})
    EventBus.$on('emittedEventCallingCrossoverMutation', this.ScatterPlotView)

    EventBus.$on('RepresentationSelectionCM', data => {this.representationDef = data})
    EventBus.$on('RepresentationSelectionCM', this.ScatterPlotView)

    EventBus.$on('Responsive', data => {
    this.WH = data})
    EventBus.$on('ResponsiveandChange', data => {
    this.WH = data})

    // reset view
    EventBus.$on('resetViews', this.reset)
  }
}
</script>