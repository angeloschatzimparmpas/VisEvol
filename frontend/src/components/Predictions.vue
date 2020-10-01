<template>
  <div id="containerForAll">
    <div id="containerAll"></div>
    <div id="containerSelection"></div>
    <div id="LegendMain"></div>
    <div id="LegendHeat"></div>
  </div>
</template>

<script>
import * as d3Base from 'd3'
import { EventBus } from '../main.js'
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
      classesNumber: 9,
      InfoPred: []
    }
  },
  methods: {
    reset () {
      var svg = d3.select("#containerAll");
      svg.selectAll("*").remove();
      var svg = d3.select("#containerSelection");
      svg.selectAll("*").remove();
      var svgLegG = d3.select("#LegendMain");
      svgLegG.selectAll("*").remove();
      var svgLeg = d3.select("#LegendHeat");
      svgLeg.selectAll("*").remove();
      this.GetResultsAll = []
      this.GetResultsSelection = []
      this.predictSelection = []
      this.StoreIndices = []
      this.InfoPred = []
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
          dataAver.push({ id: -1, value: 1.0 })
          dataKNN.push({ id: -1, value: 1.0 })
          dataLR.push({ id: -1, value: 1.0 })
          dataMLP.push({ id: -1, value: 1.0 })
          dataRF.push({ id: -1, value: 1.0 })
          dataGradB.push({ id: -1, value: 1.0 })
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
				height = 77;
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
		var groupSpacing = 42;
		var cellSpacing = 2;
    var cellSize = Math.floor((width - 1 * groupSpacing) / (13 * sqrtSize)) - cellSpacing;

		// === First call === //
		databind(classStore, size, sqrtSize); // ...then update the databind function

		var t = d3.timer(function(elapsed) {
			draw();
			if (elapsed > 2500) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize) {

			colourScale = d3.scaleSequential(d3.interpolateGreens).domain([1, 0])

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
        .attr('fillStyle', function(d) { if(d.id == -1) { return "#ffffff" } else { return colourScale(d.value);}})
        .attr('fill-opacity', function(d) { 
          if (d.id == -1) {
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
          dataAver.push({ id: -1, value: 0 })
          dataKNN.push({ id: -1, value: 0 })
          dataLR.push({ id: -1, value: 0 })
          dataMLP.push({ id: -1, value: 0 })
          dataRF.push({ id: -1, value: 0 })
          dataGradB.push({ id: -1, value: 0 })
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
				height = 77;
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
		var groupSpacing = 42;
		var cellSpacing = 2;
    var cellSize = Math.floor((width - 1 * groupSpacing) / (13 * sqrtSize)) - cellSpacing;

		// === First call === //
		databind(classStore, size, sqrtSize); // ...then update the databind function
		
		var t = d3.timer(function(elapsed) {
			draw();
			if (elapsed > 2500) t.stop();
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
        .attr('fillStyle', function(d) { if(d.id == -1) { return "#ffffff" } else { return colourScale(d.value);} })
        .attr('fill-opacity', function(d) { 
          if (d.id == -1) {
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
  legendCol () {
    //==================================================
    var viewerWidth = this.responsiveWidthHeight[0]*7
    var viewerHeight = this.responsiveWidthHeight[1]*1.6
    var viewerPosTop = viewerHeight * 0.06;
    var cellSizeHeat = 20
    var legendElementWidth = cellSizeHeat * 3;

    var info = JSON.parse(this.InfoPred[13])

    // http://bl.ocks.org/mbostock/5577023
    var colors = colorbrewer.PRGn[this.classesNumber];

    var svgLegGl = d3.select("#LegendMain");
      svgLegGl.selectAll("*").remove();

    var svgLeg = d3.select("#LegendHeat");
      svgLeg.selectAll("*").remove();

    var svgLegGl = d3.select("#LegendMain").append("svg")
      .attr("width", viewerWidth)
      .attr("height", viewerHeight*0.35)
      .style("margin-top", "0")

    var initialValue = 35
    var multiple = 105
    var heightText = 180

    svgLegGl.append("line")
      .attr("x1", 613)
      .attr("y1", 0)
      .attr("x2", 613)
      .attr("y2", heightText+30)
      .style("stroke-width", 2)
      .style("stroke", "black")
      .style("fill", "none");

    svgLegGl.append("text").attr("x", -52).attr("y", 12).text("All").style("font-size", "16px").style("font-weight", "bold").attr("alignment-baseline","top").attr("transform", 
                "rotate(-90)");
    svgLegGl.append("text").attr("x", -142).attr("y", 12).text("Sel.").style("font-size", "16px").style("font-weight", "bold").attr("alignment-baseline","top").attr("transform", 
                "rotate(-90)");

    svgLegGl.append("text").attr("x", initialValue).attr("y", heightText).text("Mean").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*1).attr("y", heightText).text("KNN").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*2).attr("y", heightText).text("LR").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*3-5).attr("y", heightText).text("MLP").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*4-6).attr("y", heightText).text("RF").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*5-12).attr("y", heightText).text("GradB").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*6-16).attr("y", heightText).text("Mean").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*7-20).attr("y", heightText).text("KNN").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*8-18).attr("y", heightText).text("LR").style("font-size", "14px").attr("alignment-baseline","top")    
    svgLegGl.append("text").attr("x", initialValue+multiple*9-25).attr("y", heightText).text("MLP").style("font-size", "14px").attr("alignment-baseline","top")    
    svgLegGl.append("text").attr("x", initialValue+multiple*10-24).attr("y", heightText).text("RF").style("font-size", "14px").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", initialValue+multiple*11-36).attr("y", heightText).text("GradB").style("font-size", "14px").attr("alignment-baseline","top")
    
    svgLegGl.append("text").attr("x", 275).attr("y", heightText+30).text(info[0]).style("font-size", "16px").style("font-weight", "bold").attr("alignment-baseline","top")
    svgLegGl.append("text").attr("x", 882).attr("y", heightText+30).text(info[1]).style("font-size", "16px").style("font-weight", "bold").attr("alignment-baseline","top")

      var svgLeg = d3.select("#LegendHeat").append("svg")
        .attr("width", viewerWidth/2)
        .attr("height", viewerHeight*0.10)
        .style("margin-top", "35px")

      var legend = svgLeg.append('g')
          .attr("class", "legend")
          .attr("transform", "translate(0,0)")
          .selectAll(".legendElement")
          .data([1.00, 0.75, 0.50, 0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
          .enter().append("g")
          .attr("class", "legendElement");

      legend.append("svg:rect")
          .attr("x", function(d, i) {
              return (legendElementWidth * i) + 50;
          })
          .attr("y", viewerPosTop)
          .attr("class", "cellLegend bordered")
          .attr("width", legendElementWidth)
          .attr("height", cellSizeHeat / 2)
          .style("fill", function(d, i) {
              return colors[i];
          });

      legend.append("text")
          .attr("class", "mono legendElement")
          .text(function(d, i) {
            if (i < 4) {
              return "-" + (d * 100) + "%";
            } else if (i > 4) {
              return "+" + (d * 100) + "%";
            } else {
              return "" + (d * 100) + "%";
            }

          })
          .attr("x", function(d, i) {
            if (i > 4) {
              return (legendElementWidth * i) + 60;
            } else if (i == 4) {
              return (legendElementWidth * i) + 72;
            } else {
              return (legendElementWidth * i) + 52;
            }
              
          })
          .attr("y", (viewerPosTop + cellSizeHeat) + 5);

      svgLeg.append("text").attr("x", 220).attr("y", 30).text("Difference in predictive power").style("font-size", "16px").attr("alignment-baseline","top")
  },
  },
  mounted () {
      EventBus.$on('emittedEventCallingInfo', data => { this.InfoPred = data })
      EventBus.$on('LegendPredict', this.legendCol)

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

    #containerForAll {
      height: 100px;
      position: relative;
    }
    #LegendMain {
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }
    #LegendMain {
      z-index: 10;
    }
</style>