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
    }
  },
  methods: {
    reset () {
      var svg = d3.select("#containerAll");
      svg.selectAll("*").remove();
      var svg = d3.select("#containerSelection");
      svg.selectAll("*").remove();
    },
    Grid () {

      Array.prototype.multiIndexOf = function (el) { 
          var idxs = [];
          for (var i = this.length - 1; i >= 0; i--) {
              if (this[i] === el) {
                  idxs.unshift(i);
              }
          }
          return idxs;
      };
      // Clear Heatmap first
      var svg = d3.select("#GridView");
      svg.selectAll("*").remove();

      var yValues = JSON.parse(this.GetResultsAll[6])
      var targetNames = JSON.parse(this.GetResultsAll[7])

      var getIndices = []
      for (let i = 0; i < targetNames.length; i++) {
        getIndices.push(yValues.multiIndexOf(targetNames[i]))
      }
      getIndices.reverse()

      var predictions = JSON.parse(this.GetResultsAll[12])
      var KNNPred = predictions[0]
      var LRPred = predictions[1]
      var PredAver = predictions[2]

      var dataAver = []
      var dataAverGetResults = []
      var dataKNN = []
      var dataKNNResults = []
      var dataLR = []
      var dataLRResults = []

      var max = 0
      for (let i = 0; i < targetNames.length; i++) {
        if (getIndices[targetNames[i]].length > max) {
          max = getIndices[targetNames[i]].length
        } 
      }

      var sqrtSize = Math.ceil(Math.sqrt(max))
      var size = sqrtSize * sqrtSize

      for (let i = 0; i < targetNames.length; i++) {
        dataAver = [];
        dataKNN = []
        dataLR = []
        getIndices[targetNames[i]].forEach(element => {
          dataAver.push({ id: element, value: PredAver[element][targetNames[i]] })
          dataKNN.push({ id: element, value: KNNPred[element][targetNames[i]] })
          dataLR.push({ id: element, value: LRPred[element][targetNames[i]] })
        });
        for (let j = 0; j < size - getIndices[targetNames[i]].length; j++) {
          dataAver.push({ id: null, value: 0.0 })
          dataKNN.push({ id: null, value: 0.0 })
          dataLR.push({ id: null, value: 0.0 })
        }
        dataAverGetResults.push(dataAver)
        dataKNNResults.push(dataKNN)
        dataLRResults.push(dataLR)
      }
    dataAverGetResults.reverse()
    dataKNNResults.reverse()
    dataLRResults.reverse()
    
    var classArray = []

    for (let i = 0; i < dataAverGetResults.length; i++) {
      dataAverGetResults[i].sort((a, b) => (a.value > b.value) ? -1 : 1)
      var len = dataAverGetResults[i].length
      var indices = new Array(len)
      for (let j = 0; j < len; j++) {
        indices[j] = dataAverGetResults[i][j].id;
      }
      
      dataKNNResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataLRResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      classArray.push(dataAverGetResults[i].concat(dataKNNResults[i], dataLRResults[i]));
    }
    
    var classStore = [].concat.apply([], classArray);

		// === Set up canvas === //

		var width = 2500,
				height = 125;
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
		var groupSpacing = 60;
		var cellSpacing = 2;
    var cellSize = Math.floor((width - 11 * groupSpacing) / (15.2*sqrtSize)) - cellSpacing;

		// === First call === //
		databind(classStore, size, sqrtSize); // ...then update the databind function
		
		var t = d3.timer(function(elapsed) {
			draw();
			if (elapsed > 300) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize) {

      
			colourScale = d3.scaleSequential(d3.interpolateReds).domain(d3.extent(data, function(d) { return d.value; }));

			var join = custom.selectAll('custom.rect')
        .data(data);
        
			var enterSel = join.enter()
				.append('custom')
				.attr('class', 'rect')
	      .attr('x', function(d, i) {
	        var x0 = Math.floor(i / 169) % 13, x1 = Math.floor(i % 13);
	        return groupSpacing * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10);
	      })
	      .attr('y', function(d, i) {
	        var y0 = Math.floor(i / 1000), y1 = Math.floor(i % 169 / 13);
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

      Array.prototype.multiIndexOf = function (el) { 
          var idxs = [];
          for (var i = this.length - 1; i >= 0; i--) {
              if (this[i] === el) {
                  idxs.unshift(i);
              }
          }
          return idxs;
      };
      // Clear Heatmap first
      var svg = d3.select("#GridView");
      svg.selectAll("*").remove();

      var yValues = JSON.parse(this.GetResultsSelection[6])
      var targetNames = JSON.parse(this.GetResultsSelection[7])

      var getIndices = []
      for (let i = 0; i < targetNames.length; i++) {
        getIndices.push(yValues.multiIndexOf(targetNames[i]))
      }
      getIndices.reverse()

      var predictions = JSON.parse(this.GetResultsSelection[12])
      var KNNPred = predictions[0]
      var LRPred = predictions[1]
      var PredAver = predictions[2]

      var dataAver = []
      var dataAverGetResults = []
      var dataKNN = []
      var dataKNNResults = []
      var dataLR = []
      var dataLRResults = []

      var max = 0
      for (let i = 0; i < targetNames.length; i++) {
        if (getIndices[targetNames[i]].length > max) {
          max = getIndices[targetNames[i]].length
        } 
      }

      var sqrtSize = Math.ceil(Math.sqrt(max))
      var size = sqrtSize * sqrtSize

      for (let i = 0; i < targetNames.length; i++) {
        dataAver = [];
        dataKNN = []
        dataLR = []
        getIndices[targetNames[i]].forEach(element => {
          dataAver.push({ id: element, value: PredAver[element][targetNames[i]] })
          dataKNN.push({ id: element, value: KNNPred[element][targetNames[i]] })
          dataLR.push({ id: element, value: LRPred[element][targetNames[i]] })
        });
        for (let j = 0; j < size - getIndices[targetNames[i]].length; j++) {
          dataAver.push({ id: null, value: 0.0 })
          dataKNN.push({ id: null, value: 0.0 })
          dataLR.push({ id: null, value: 0.0 })
        }
        dataAverGetResults.push(dataAver)
        dataKNNResults.push(dataKNN)
        dataLRResults.push(dataLR)
      }
    dataAverGetResults.reverse()
    dataKNNResults.reverse()
    dataLRResults.reverse()
    
    var classArray = []

    for (let i = 0; i < dataAverGetResults.length; i++) {
      dataAverGetResults[i].sort((a, b) => (a.value > b.value) ? -1 : 1)
      var len = dataAverGetResults[i].length
      var indices = new Array(len)
      for (let j = 0; j < len; j++) {
        indices[j] = dataAverGetResults[i][j].id;
      }
      
      dataKNNResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      dataLRResults[i].sort(function(a, b){
        return indices.indexOf(a.id) - indices.indexOf(b.id)
      });

      classArray.push(dataAverGetResults[i].concat(dataKNNResults[i], dataLRResults[i]));
    }
    
    var classStore = [].concat.apply([], classArray);

		// === Set up canvas === //

		var width = 2500,
				height = 125;
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
		var groupSpacing = 60;
		var cellSpacing = 2;
    var cellSize = Math.floor((width - 11 * groupSpacing) / (15.2*sqrtSize)) - cellSpacing;

		// === First call === //
		databind(classStore, size, sqrtSize); // ...then update the databind function
		
		var t = d3.timer(function(elapsed) {
			draw();
			if (elapsed > 300) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize) {

      
			colourScale = d3.scaleSequential(d3.interpolateReds).domain(d3.extent(data, function(d) { return d.value; }));

			var join = custom.selectAll('custom.rect')
        .data(data);
        
			var enterSel = join.enter()
				.append('custom')
				.attr('class', 'rect')
	      .attr('x', function(d, i) {
	        var x0 = Math.floor(i / 169) % 13, x1 = Math.floor(i % 13);
	        return groupSpacing * x0 + (cellSpacing + cellSize) * (x1 + x0 * 10);
	      })
	      .attr('y', function(d, i) {
	        var y0 = Math.floor(i / 1000), y1 = Math.floor(i % 169 / 13);
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