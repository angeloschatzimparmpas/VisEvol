<template>
  <div>
    <div id="containerAll"></div>
    <div id="containerSelection"></div>
  </div>
</template>

<script>
import * as d3Base from 'd3'
import { EventBus } from '../main.js'
import $ from 'jquery'
import * as colorbr from 'colorbrewer'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)
const colorbrewer = Object.assign(colorbr)

export default {
  name: "Predictions",
  data () {
    return {
      GetResultsAll: [],
      GetResultsSelection: [],
      responsiveWidthHeight: [],
      predictSelection: [],
      StoreIndices: [],
    }
  },
  methods: {
    reset () {
      var svg = d3.select("#containerAll");
      svg.selectAll("*").remove();
      var svg = d3.select("#containerSelection");
      svg.selectAll("*").remove();
      this.GetResultsAll = []
      this.GetResultsSelection = []
      this.predictSelection = []
      this.StoreIndices = []
    },
    Grid () {

      var svg = d3.select("#containerAll");
      svg.selectAll("*").remove();

      var yValues = JSON.parse(this.GetResultsAll[6])
      var targetNames = JSON.parse(this.GetResultsAll[7])

      var getIndices = []
      for (let i = 0; i < targetNames.length; i++) {
        let clTemp = []
        let j = -1
        while((j = yValues.indexOf(targetNames[i], j + 1)) !== -1) {
          clTemp.push(j);
        }
        getIndices.push(clTemp)
      }
      getIndices.reverse()

      var predictions = JSON.parse(this.GetResultsAll[12])
      var KNNPred = predictions[0]
      var LRPred = predictions[1]
      var MLPPred = predictions[2]
      var RFPred = predictions[3]
      var GradBPred = predictions[4]
      var PredAver = predictions[5]
      var dataAver = []
      var dataAverGetResults = []
      var dataKNN = []
      var dataKNNResults = []
      var dataLR = []
      var dataLRResults = []
      var dataMLP = []
      var dataMLPResults = []
      var dataRF = []
      var dataRFResults = []
      var dataGradB = []
      var dataGradBResults = []

      var max = 0
      for (let i = 0; i < targetNames.length; i++) {
        if (getIndices[targetNames[i]].length > max) {
          max = getIndices[targetNames[i]].length
        } 
      }

      var sqrtSize = Math.ceil(Math.sqrt(max))
      var size = sqrtSize * sqrtSize

      for (let i = 0; i < targetNames.length; i++) {
        dataAver = []
        dataKNN = []
        dataLR = []
        dataMLP = []
        dataRF = []
        dataGradB = []
        getIndices[targetNames[i]].forEach(element => {
          dataAver.push({ id: element, value: PredAver[element][targetNames[i]] })
          dataKNN.push({ id: element, value: KNNPred[element][targetNames[i]] })
          dataLR.push({ id: element, value: LRPred[element][targetNames[i]] })
          dataMLP.push({ id: element, value: MLPPred[element][targetNames[i]] })
          dataRF.push({ id: element, value: RFPred[element][targetNames[i]] })
          dataGradB.push({ id: element, value: GradBPred[element][targetNames[i]] })
        });
        for (let j = 0; j < size - getIndices[targetNames[i]].length; j++) {
          dataAver.push({ id: null, value: 1.0 })
          dataKNN.push({ id: null, value: 1.0 })
          dataLR.push({ id: null, value: 1.0 })
          dataMLP.push({ id: null, value: 1.0 })
          dataRF.push({ id: null, value: 1.0 })
          dataGradB.push({ id: null, value: 1.0 })
        }
        dataAverGetResults.push(dataAver)
        dataKNNResults.push(dataKNN)
        dataLRResults.push(dataLR)
        dataMLPResults.push(dataMLP)
        dataRFResults.push(dataRF)
        dataGradBResults.push(dataGradB)
      }
    dataAverGetResults.reverse()
    dataKNNResults.reverse()
    dataLRResults.reverse()
    dataMLPResults.reverse()
    dataRFResults.reverse()
    dataGradBResults.reverse()
    
    var classArray = []
    this.StoreIndices = []
    for (let i = 0; i < dataAverGetResults.length; i++) {
      dataAverGetResults[i].sort((a, b) => (a.value > b.value) ? 1 : -1)
      var len = dataAverGetResults[i].length
      var indices = new Array(len)
      for (let j = 0; j < len; j++) {
        indices[j] = dataAverGetResults[i][j].id;
      }
      this.StoreIndices.push(indices)
      
      dataKNNResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataLRResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataMLPResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataRFResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataGradBResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      classArray.push(dataAverGetResults[i].concat(dataKNNResults[i], dataLRResults[i],dataMLPResults[i],dataRFResults[i],dataGradBResults[i]));
    }

    var classStore = [].concat.apply([], classArray);

		// === Set up canvas === //

		var width = 1200,
				height = 85;
		var colourScale;


		var canvas = d3.select('#containerAll')
			.append('canvas')
			.attr('width', width)
			.attr('height', height);

		var context = canvas.node().getContext('2d');

		// === Bind data to custom elements === //

		var customBase = document.createElement('custom');
		var custom = d3.select(customBase); // this is our svg replacement

    // settings for a grid with 40 cells in a row and 2x5 cells in a group
		var groupSpacing = 40;
		var cellSpacing = 2;
    var cellSize = Math.floor((width - 1 * groupSpacing) / (13 * sqrtSize)) - cellSpacing;

		// === First call === //
		databind(classStore, size, sqrtSize); // ...then update the databind function
		
		var t = d3.timer(function(elapsed) {
			draw();
			if (elapsed > 1000) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize) {

			colourScale = d3.scaleSequential(d3.interpolateReds).domain([1, 0])

			var join = custom.selectAll('custom.rect')
        .data(data);
        
			var enterSel = join.enter()
				.append('custom')
				.attr('class', 'rect')
	      .attr('x', function(d, i) {
	        var x0 = Math.floor(i / size) % sqrtSize, x1 = Math.floor(i % sqrtSize);
	        return groupSpacing * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10);
	      })
	      .attr('y', function(d, i) {
	        var y0 = Math.floor(i / data.length), y1 = Math.floor(i % size / sqrtSize);
	        return groupSpacing * y0 + (cellSpacing + cellSize) * (y1 + y0 * 10);
	      })
				.attr('width', 0)
				.attr('height', 0);

			join
				.merge(enterSel)
				.transition()
				.attr('width', cellSize)
				.attr('height', cellSize)
        .attr('fillStyle', function(d) { return colourScale(d.value); })
        .attr('fill-opacity', function(d) { 
          if (d.id == null) {
            return "0.0";
          } else {
            return "1.0"; 
          } 
        });

			var exitSel = join.exit()
				.transition()
				.attr('width', 0)
				.attr('height', 0)
				.remove();

		} // databind()


		// === Draw canvas === //

		function draw() {

			// clear canvas
			
			context.fillStyle = '#fff';
			context.fillRect(0, 0, width, height);

			
			// draw each individual custom element with their properties
			
			var elements = custom.selectAll('custom.rect') // this is the same as the join variable, but used here to draw
			
			elements.each(function(d,i) {

				// for each virtual/custom element...

				var node = d3.select(this);
				context.fillStyle = node.attr('fillStyle');
				context.fillRect(node.attr('x'), node.attr('y'), node.attr('width'), node.attr('height'))

			});

		} // draw()

  },
  GridSelection () {

      var svg = d3.select("#containerSelection");
      svg.selectAll("*").remove();

      var predictionsAll = JSON.parse(this.GetResultsAll[12])

      if (this.predictSelection.length != 0) {
        var predictions = this.predictSelection
        var KNNPred = predictions[0]
        var LRPred = predictions[1]
        var MLPPred = predictions[2]
        var RFPred = predictions[3]
        var GradBPred = predictions[4]
        var PredAver = predictions[5]
      } else {
        var KNNPred = predictionsAll[0]
        var LRPred = predictionsAll[1]
        var MLPPred = predictionsAll[2]
        var RFPred = predictionsAll[3]
        var GradBPred = predictionsAll[4]
        var PredAver = predictionsAll[5]
      }
      var KNNPredAll = predictionsAll[0]
      var LRPredAll = predictionsAll[1]
      var MLPPredAll = predictionsAll[2]
      var RFPredAll = predictionsAll[3]
      var GradBPredAll = predictionsAll[4]
      var PredAverAll = predictionsAll[5]

      var yValues = JSON.parse(this.GetResultsSelection[6])
      var targetNames = JSON.parse(this.GetResultsSelection[7])

      var getIndices = []
      for (let i = 0; i < targetNames.length; i++) {
        let clTemp = []
        let j = -1
        while((j = yValues.indexOf(targetNames[i], j + 1)) !== -1) {
          clTemp.push(j);
        }
        getIndices.push(clTemp)
      }
      getIndices.reverse()

      var dataAver = []
      var dataAverGetResults = []
      var dataKNN = []
      var dataKNNResults = []
      var dataLR = []
      var dataLRResults = []
      var dataMLP = []
      var dataMLPResults = []
      var dataRF = []
      var dataRFResults = []
      var dataGradB = []
      var dataGradBResults = []

      var max = 0
      for (let i = 0; i < targetNames.length; i++) {
        if (getIndices[targetNames[i]].length > max) {
          max = getIndices[targetNames[i]].length
        } 
      }

      var sqrtSize = Math.ceil(Math.sqrt(max))
      var size = sqrtSize * sqrtSize

      for (let i = 0; i < targetNames.length; i++) {
        dataAver = []
        dataKNN = []
        dataLR = []
        dataMLP = []
        dataRF = []
        dataGradB = []
        getIndices[targetNames[i]].forEach(element => {
          dataAver.push({ id: element, value: PredAver[element][targetNames[i]] - PredAverAll[element][targetNames[i]] })
          dataKNN.push({ id: element, value: KNNPred[element][targetNames[i]] - KNNPredAll[element][targetNames[i]] })
          dataLR.push({ id: element, value: LRPred[element][targetNames[i]] - LRPredAll[element][targetNames[i]] })
          dataMLP.push({ id: element, value: MLPPred[element][targetNames[i]] - MLPPredAll[element][targetNames[i]] })
          dataRF.push({ id: element, value: RFPred[element][targetNames[i]] - RFPredAll[element][targetNames[i]] })
          dataGradB.push({ id: element, value: GradBPred[element][targetNames[i]] - GradBPredAll[element][targetNames[i]] })
        });
        for (let j = 0; j < size - getIndices[targetNames[i]].length; j++) {
          dataAver.push({ id: null, value: 0 })
          dataKNN.push({ id: null, value: 0 })
          dataLR.push({ id: null, value: 0 })
          dataMLP.push({ id: null, value: 0 })
          dataRF.push({ id: null, value: 0 })
          dataGradB.push({ id: null, value: 0 })
        }
        dataAverGetResults.push(dataAver)
        dataKNNResults.push(dataKNN)
        dataLRResults.push(dataLR)
        dataMLPResults.push(dataMLP)
        dataRFResults.push(dataRF)
        dataGradBResults.push(dataGradB)
      }
    dataAverGetResults.reverse()
    dataKNNResults.reverse()
    dataLRResults.reverse()
    dataMLPResults.reverse()
    dataRFResults.reverse()
    dataGradBResults.reverse()

    var classArray = []

    for (let i = 0; i < dataAverGetResults.length; i++) {
      
      var indices = this.StoreIndices[i]
      dataAverGetResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataKNNResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataLRResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataMLPResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataRFResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataGradBResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      classArray.push(dataAverGetResults[i].concat(dataKNNResults[i], dataLRResults[i], dataMLPResults[i], dataRFResults[i], dataGradBResults[i]));
    }
    
    var classStore = [].concat.apply([], classArray);

		// === Set up canvas === //

		var width = 1200,
				height = 85;
		var colourScale;


		var canvas = d3.select('#containerSelection')
			.append('canvas')
			.attr('width', width)
			.attr('height', height);

		var context = canvas.node().getContext('2d');

		// === Bind data to custom elements === //

		var customBase = document.createElement('custom');
		var custom = d3.select(customBase); // this is our svg replacement

    // settings for a grid with 40 cells in a row and 2x5 cells in a group
		var groupSpacing = 40;
		var cellSpacing = 2;
    var cellSize = Math.floor((width - 1 * groupSpacing) / (13 * sqrtSize)) - cellSpacing;

		// === First call === //
		databind(classStore, size, sqrtSize); // ...then update the databind function
		
		var t = d3.timer(function(elapsed) {
			draw();
			if (elapsed > 1000) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize) {

      
			colourScale = d3.scaleSequential(d3.interpolatePRGn).domain([-1, 1])

			var join = custom.selectAll('custom.rect')
        .data(data);
        
			var enterSel = join.enter()
				.append('custom')
				.attr('class', 'rect')
	      .attr('x', function(d, i) {
	        var x0 = Math.floor(i / size) % sqrtSize, x1 = Math.floor(i % sqrtSize);
	        return groupSpacing * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10);
	      })
	      .attr('y', function(d, i) {
	        var y0 = Math.floor(i / data.length), y1 = Math.floor(i % size / sqrtSize);
	        return groupSpacing * y0 + (cellSpacing + cellSize) * (y1 + y0 * 10);
	      })
				.attr('width', 0)
				.attr('height', 0);

			join
				.merge(enterSel)
				.transition()
				.attr('width', cellSize)
				.attr('height', cellSize)
        .attr('fillStyle', function(d) { return colourScale(d.value); })
        .attr('fill-opacity', function(d) { 
          if (d.id == null) {
            return "0.0";
          } else {
            return "1.0"; 
          } 
        });

			var exitSel = join.exit()
				.transition()
				.attr('width', 0)
				.attr('height', 0)
				.remove();

		} // databind()


		// === Draw canvas === //

		function draw() {

			// clear canvas
			
			context.fillStyle = '#fff';
			context.fillRect(0, 0, width, height);

			
			// draw each individual custom element with their properties
			
			var elements = custom.selectAll('custom.rect') // this is the same as the join variable, but used here to draw
			
			elements.each(function(d,i) {

				// for each virtual/custom element...

				var node = d3.select(this);
				context.fillStyle = node.attr('fillStyle');
				context.fillRect(node.attr('x'), node.attr('y'), node.attr('width'), node.attr('height'))

			});

		} // draw()

  },
  },
  mounted () {
      EventBus.$on('emittedEventCallingGrid', data => { this.GetResultsAll = data; })
      EventBus.$on('emittedEventCallingGrid', this.Grid)

      EventBus.$on('emittedEventCallingGridSelection', data => { this.GetResultsSelection = data; })
      EventBus.$on('emittedEventCallingGridSelection', this.GridSelection)

      EventBus.$on('SendSelectedPointsToServerEvent', data => { this.predictSelection = data; })
      EventBus.$on('SendSelectedPointsToServerEvent', this.GridSelection)

      EventBus.$on('Responsive', data => {
      this.responsiveWidthHeight = data})
      EventBus.$on('ResponsiveandChange', data => {
      this.responsiveWidthHeight = data})

      // reset the views
      EventBus.$on('resetViews', this.reset)
    }
}
</script>

<style type="text/css">
		canvas {
			border:  1px dotted #ccc;
		}
</style>