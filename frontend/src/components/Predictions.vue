<template>
  <div id="containerForAll">
    <div id="containerAll"></div>
    <div id="containerSelection"></div>
    <div id="LegendMain"></div>
    <b-row>
      <b-col cols="2">
        <div id="HistClass0" style = "margin-top: 42px"></div>
      </b-col>
      <b-col cols="8">
        <div id="LegendHeat"></div>
      </b-col>
      <b-col cols="2" style="margin-left: -120px !important">
        <div id="HistClass1" style = "margin-top: 42px"></div>
      </b-col>
    </b-row>
  </div>
</template>

<script>
import * as d3Base from 'd3'
import { EventBus } from '../main.js'
import * as colorbr from 'colorbrewer'
import * as Plotly from 'plotly.js'

// attach all d3 plugins to the d3 library
const d3v5 = Object.assign(d3Base)
const colorbrewer = Object.assign(colorbr)

export default {
  name: "Predictions",
  data () {
    return {
      GetResultsAll: [],
      responsiveWidthHeight: [],
      predictSelection: [],
      StoreIndices: [],
      classesNumber: 9,
      InfoPred: [],
      flag: false,
      classOver0: [],
      classOver1: [],
      RetrieveValueFi: 'biodegC' // default file name
    }
  },
  methods: {
    reset () {
      var svg = d3v5.select("#containerAll");
      svg.selectAll("*").remove();
      var svg = d3v5.select("#containerSelection");
      svg.selectAll("*").remove();
      var svgLegG = d3v5.select("#LegendMain");
      svgLegG.selectAll("*").remove();
      var svgLeg = d3v5.select("#LegendHeat");
      svgLeg.selectAll("*").remove();
      Plotly.purge('HistClass0')
      Plotly.purge('HistClass1')
      this.GetResultsAll = []
      this.predictSelection = []
      this.StoreIndices = []
      this.InfoPred = []
    },
    Grid () {

      var svg = d3v5.select("#containerAll");
      svg.selectAll("*").remove();

      var yValues = JSON.parse(this.GetResultsAll[14])

      var targetNames = JSON.parse(this.GetResultsAll[7])
      var getIndices = []

      var predictions = JSON.parse(this.GetResultsAll[12])
      var KNNPred = predictions[0]
      var LRPred = predictions[1]
      var MLPPred = predictions[2]
      var RFPred = predictions[3]
      var GradBPred = predictions[4]
      var PredAver = predictions[5]

      if (!this.flag) {
        for (let i = 0; i < targetNames.length; i++) {
          let clTemp = []
          let j = -1
          while((j = yValues.indexOf(targetNames[i], j + 1)) !== -1) {
            clTemp.push(j);
          }
          getIndices.push(clTemp)
        }
        var Class0 = []
        var Class1 = []
      }
      else {
        var tempFirst = []
        for (let i = 0; i < 100; i++) {
          tempFirst.push(i)
        }
        var tempLast = []
        for (let i = 100; i < 200; i++) {
          tempLast.push(i)
        }
        getIndices.push(tempFirst)
        getIndices.push(tempLast)
        this.classOver0 = predictions[6]
        this.classOver1 = predictions[7]
      }
      if (this.RetrieveValueFi == "heartC") {
        getIndices.reverse()
      }

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

      var sqrtSize = 13
      var size = 169

      for (let i = 0; i < targetNames.length; i++) {
        dataAver = []
        dataKNN = []
        dataLR = []
        dataMLP = []
        dataRF = []
        dataGradB = []
        getIndices[targetNames[i]].forEach(element => {
          if (PredAver.length == 0) {
            dataAver.push({ id: -1, value: 0 })
          } else {
            dataAver.push({ id: element, value: PredAver[element] })
          }
          if (KNNPred.length == 0) {
            dataKNN.push({ id: -1, value: 0 })
          } else {
            dataKNN.push({ id: element, value: KNNPred[element] })
          }
          if (LRPred.length == 0) {
            dataLR.push({ id: -1, value: 0 })
          } else {
            dataLR.push({ id: element, value: LRPred[element] })
          }
          if (MLPPred.length == 0) {
            dataMLP.push({ id: -1, value: 0 })
          } else {
            dataMLP.push({ id: element, value: MLPPred[element] })
          }
          if (RFPred.length == 0) {
            dataRF.push({ id: -1, value: 0 })
          } else {
            dataRF.push({ id: element, value: RFPred[element] })
          }
          if (GradBPred.length == 0) {
            dataGradB.push({ id: -1, value: 0 })
          } else {
            dataGradB.push({ id: element, value: GradBPred[element] })
          } 
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
    // dataAverGetResults.reverse()
    // dataKNNResults.reverse()
    // dataLRResults.reverse()
    // dataMLPResults.reverse()
    // dataRFResults.reverse()
    // dataGradBResults.reverse()

    var classArray = []
    this.StoreIndices = []
    for (let i = 0; i < dataAverGetResults.length; i++) {
      dataAverGetResults[i].sort((a, b) => (a.value > b.value) ? -1 : 1)
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


		var canvas = d3v5.select('#containerAll')
			.append('canvas')
			.attr('width', width)
      .attr('height', height);

		var context = canvas.node().getContext('2d');

		// === Bind data to custom elements === //

		var customBase = document.createElement('custom');
		var custom = d3v5.select(customBase); // this is our svg replacement

    // settings for a grid with 40 cells in a row and 2x5 cells in a group
		var groupSpacing = 42;
		var cellSpacing = 2;
    var cellSize = 4

    if (!this.flag) {
      var lengthOverall = classStore.length
    } else {
      var lengthOverall = 2028
    }
		// === First call === //
		databind(classStore, size, sqrtSize, lengthOverall); // ...then update the databind function

		var t = d3v5.timer(function(elapsed) {
			draw();
			if (elapsed > 2500) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize, lengthOverallLocal) {

			colourScale = d3v5.scaleSequential(d3v5.interpolateGreens).domain([0, 100])

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
	        var y0 = Math.floor(i / lengthOverallLocal), y1 = Math.floor(i % size / sqrtSize);
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

				var node = d3v5.select(this);
				context.fillStyle = node.attr('fillStyle');
				context.fillRect(node.attr('x'), node.attr('y'), node.attr('width'), node.attr('height'))

			});

		} // draw()

  },
  GridSelection () {

      var svg = d3v5.select("#containerSelection");
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

      var yValues = JSON.parse(this.GetResultsAll[14])
      var targetNames = JSON.parse(this.GetResultsAll[7])

      var getIndices = []
      if (!this.flag) {

        for (let i = 0; i < targetNames.length; i++) {
          let clTemp = []
          let j = -1
          while((j = yValues.indexOf(targetNames[i], j + 1)) !== -1) {
            clTemp.push(j);
          }
          getIndices.push(clTemp)
        }

      }
      else {
        var tempFirst = []
        for (let i = 0; i < 100; i++) {
          tempFirst.push(i)
        }
        var tempLast = []
        for (let i = 100; i < 200; i++) {
          tempLast.push(i)
        }
        getIndices.push(tempFirst)
        getIndices.push(tempLast)
      }
      
      if (this.RetrieveValueFi == "heartC") {
        getIndices.reverse()
      }

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

      var sqrtSize = 13
      var size = 169

      for (let i = 0; i < targetNames.length; i++) {
        dataAver = []
        dataKNN = []
        dataLR = []
        dataMLP = []
        dataRF = []
        dataGradB = []

        getIndices[targetNames[i]].forEach(element => {
          if (PredAver.length == 0) {
            dataAver.push({ id: -1, value: 0 })
          } else {
            dataAver.push({ id: element, value: PredAver[element] - PredAverAll[element] })
          }
          if (KNNPred.length == 0) {
            dataKNN.push({ id: -1, value: 0 })
          } else {
            dataKNN.push({ id: element, value: KNNPred[element] - KNNPredAll[element] })
          }
          if (LRPred.length == 0) {
            dataLR.push({ id: -1, value: 0 })
          } else {
 
            dataLR.push({ id: element, value: LRPred[element] - LRPredAll[element] })
          }
          if (MLPPred.length == 0) {
            dataMLP.push({ id: -1, value: 0 })
          } else {
            dataMLP.push({ id: element, value: MLPPred[element] - MLPPredAll[element] })
          }
          if (RFPred.length == 0) {
            dataRF.push({ id: -1, value: 0 })
          } else {
            dataRF.push({ id: element, value: RFPred[element] - RFPredAll[element] })
          }
          if (GradBPred.length == 0) {
            dataGradB.push({ id: -1, value: 0 })
          } else {
            dataGradB.push({ id: element, value: GradBPred[element] - GradBPredAll[element] })
          }       
          
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


		var canvas = d3v5.select('#containerSelection')
			.append('canvas')
			.attr('width', width)
			.attr('height', height);

		var context = canvas.node().getContext('2d');

		// === Bind data to custom elements === //

		var customBase = document.createElement('custom');
		var custom = d3v5.select(customBase); // this is our svg replacement

    // settings for a grid with 40 cells in a row and 2x5 cells in a group
		var groupSpacing = 42;
		var cellSpacing = 2;
    var cellSize = 4

    if (!this.flag) {
      var lengthOverall = classStore.length
    } else {
      var lengthOverall = 2028
    }

		// === First call === //
		databind(classStore, size, sqrtSize, lengthOverall); // ...then update the databind function
		
		var t = d3v5.timer(function(elapsed) {
			draw();
			if (elapsed > 2500) t.stop();
		}); // start a timer that runs the draw function for 500 ms (this needs to be higher than the transition in the databind function)


		// === Bind and draw functions === //

		function databind(data, size, sqrtSize, lengthOverallLocal) {

			colourScale = d3v5.scaleSequential(d3v5.interpolatePRGn).domain([-100, 100])

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
	        var y0 = Math.floor(i / lengthOverallLocal), y1 = Math.floor(i % size / sqrtSize);
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

				var node = d3v5.select(this);
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

    var svgLegGl = d3v5.select("#LegendMain");
      svgLegGl.selectAll("*").remove();

    var svgLeg = d3v5.select("#LegendHeat");
      svgLeg.selectAll("*").remove();

    var svgLegGl = d3v5.select("#LegendMain").append("svg")
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

      var svgLeg = d3v5.select("#LegendHeat").append("svg")
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
      
      Plotly.purge('HistClass0')
      Plotly.purge('HistClass1')

      if (this.classOver0 !=0 && this.classOver1 != 0) {
        function range(start, end) {
          var ans = [];
          for (let i = start; i < end; i++) {
              ans.push(i+1);
          }
          return ans;
        }
        var max_of_array = Math.max.apply(Math, this.classOver0)
        var min_of_array = Math.min.apply(Math, this.classOver0)
        var class0Local = Object.values(this.classOver0)
        var class1Local = Object.values(this.classOver1)

        var indicesClass0 = this.StoreIndices[0].map( function(value) { 
            return value - 100; 
        } );
        var indicesClass1 = this.StoreIndices[1]

        var indicesClass0Trim = indicesClass0.slice(0,100)
        var indicesClass1Trim = indicesClass1.slice(0,100)

        var histogram0 = []
        var histogram1 = []

        indicesClass0Trim.forEach(element => {
          histogram0.push(class0Local[element])
        });
        indicesClass1Trim.forEach(element => {
          histogram1.push(class1Local[element])    
        });

        var trace = {
          x: range(0,100),
          y: histogram0, 
          marker: {
            color: "rgba(192, 192, 192, 1)", 
            line: {
              color:  "rgba(85, 85, 85, 1)", 
              width: 1
            }
          },  
          type: "bar", 
        };

        var layout = {
          width: 300,
          height: 62,
          bargap: 0.1,
          showlegend: false,
          margin: {
            l: 40,
            r: 15,
            b: 14,
            t: 10,
            pad: 0
          },
          title: "Distribution of Instances (Sorted)",
          yaxis: {title: "Instances"}
        };
        var data = [trace]
        var config = {'displayModeBar': false}

        Plotly.newPlot('HistClass0', data, layout, config);

        var max_of_array = Math.max.apply(Math, this.classOver1);
        var min_of_array = Math.min.apply(Math, this.classOver1);

        var trace = {
          x: range(0,100),
          y: histogram1,
          marker: {
            color: "rgba(192, 192, 192, 1)", 
            line: {
              color:  "rgba(85, 85, 85, 1)", 
              width: 1
            }
          },  
          type: "bar", 
        };

        var layout = {
          width: 300,
          height: 62,
          bargap: 0.1,
          showlegend: false,
          margin: {
            l: 40,
            r: 15,
            b: 14,
            t: 10,
            pad: 0
          },
          title: "Distribution of Instances (Sorted)",
          yaxis: {title: "Instances"}
        };
        var data = [trace]
        var config = {'displayModeBar': false}

        Plotly.newPlot('HistClass1', data, layout, config);
      }
     
  },
  },
  mounted () {
      EventBus.$on('SendToServerDataSetConfirmation', data => { this.RetrieveValueFi = data })
      EventBus.$on('ON', data => {this.flag = data})

      EventBus.$on('emittedEventCallingInfo', data => { this.InfoPred = data })
      EventBus.$on('LegendPredict', this.legendCol)

      EventBus.$on('emittedEventCallingGrid', data => { this.GetResultsAll = data; })
      EventBus.$on('emittedEventCallingGrid', this.Grid)

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

.gtitle {
  transform: translate(-432px, -99px) !important;
}
</style>