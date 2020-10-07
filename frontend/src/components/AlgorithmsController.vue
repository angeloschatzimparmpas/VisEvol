<template>
  <div id="containerForAllAlg">
    <div id="Bees" class="chart-wrapper" style="min-height: 307px;"></div>
    <div id="MainPlot"></div>
    <div id="uncertainty"></div>
  </div>
</template>

<script>
import * as Plotly from 'plotly.js'
import { EventBus } from '../main.js'
import * as d3Base from 'd3'

// attach all d3 plugins to the d3 library
const d3v5 = Object.assign(d3Base)

export default {
  name: "AlgorithmsController",
  data () {
    return {
      WH: [],
      PerF: [],
      PerFCM: [],
      storedEnsem: [],
      storedCM: [],
      selectedSimple: [],
      selectedEnsem: []
    }
  },
  methods: {
    reset () {
      var svg = d3.select("#Bees");
      svg.selectAll("*").remove();
      var svg = d3.select("#MainPlot");
      svg.selectAll("*").remove();
      Plotly.purge('uncertainty')
      this.PerF = []
      this.PerFCM = []
      this.storedEnsem = []
      this.storedCM = []
      this.selectedSimple = []
      this.selectedEnsem = []
    },
    BeesFun () {
      var svg = d3.select("#Bees");
      svg.selectAll("*").remove();
      var svg = d3.select("#MainPlot");
      svg.selectAll("*").remove();
      Plotly.purge('uncertainty')
      var chart1
      var data = []
      var originalPositions = []
      var belongs = []
      var newPositions = []
      var difference = []
      var maximumDiff = 0
      var minimumDiff = 0
      var average = new Array(5).fill(0);

      for (let i=0; i<this.storedCM.length; i++){
        let tempSplit = this.storedCM[i].split(/([0-9]+)/)
        if (tempSplit[0] == 'KNN' || tempSplit[0] == 'KNNC' || tempSplit[0] == 'KNNM' || tempSplit[0] == 'KNNCC' || tempSplit[0] == 'KNNCM' || tempSplit[0] == 'KNNMC' || tempSplit[0] == 'KNNMM') {
          if (this.selectedSimple.includes(this.storedCM[i])) {
            data.push({Algorithm:"KNN",value:this.PerFCM[i], size:3, sw:true})
          } else {
            data.push({Algorithm:"KNN",value:this.PerFCM[i], size:3, sw:false})
          }
        }
        else if (tempSplit[0] == 'LR' || tempSplit[0] == 'LRC' || tempSplit[0] == 'LRM' || tempSplit[0] == 'LRCC' || tempSplit[0] == 'LRCM' || tempSplit[0] == 'LRMC' || tempSplit[0] == 'LRMM') {
          if (this.selectedSimple.includes(this.storedCM[i])) {
            data.push({Algorithm:"LR",value:this.PerFCM[i], size:3, sw:true})
          } else {
            data.push({Algorithm:"LR",value:this.PerFCM[i], size:3, sw:false})
          }
        }
        else if (tempSplit[0] == 'MLP' || tempSplit[0] == 'MLPC' || tempSplit[0] == 'MLPM' || tempSplit[0] == 'MLPCC' || tempSplit[0] == 'MLPCM' || tempSplit[0] == 'MLPMC' || tempSplit[0] == 'MLPMM') {
          if (this.selectedSimple.includes(this.storedCM[i])) {
            data.push({Algorithm:"MLP",value:this.PerFCM[i], size:3, sw:true})
          } else {
            data.push({Algorithm:"MLP",value:this.PerFCM[i], size:3, sw:false})
          }
        }
        else if (tempSplit[0] == 'RF' || tempSplit[0] == 'RFC' || tempSplit[0] == 'RFM' || tempSplit[0] == 'RFCC' || tempSplit[0] == 'RFCM' || tempSplit[0] == 'RFMC' || tempSplit[0] == 'RFMM') {
          if (this.selectedSimple.includes(this.storedCM[i])) {
            data.push({Algorithm:"RF",value:this.PerFCM[i], size:3, sw:true})
          } else {
            data.push({Algorithm:"RF",value:this.PerFCM[i], size:3, sw:false})
          }
        }
        else {
          if (this.selectedSimple.includes(this.storedCM[i])) {
            data.push({Algorithm:"GradB",value:this.PerFCM[i], size:3, sw:true})
          } else {
            data.push({Algorithm:"GradB",value:this.PerFCM[i], size:3, sw:false})
          }
        }
      }

      if (this.storedEnsem.length != 0) {
        var mergedStoreEnsembleLoc = [].concat.apply([], this.storedEnsem)
        for (let i=0; i<mergedStoreEnsembleLoc.length; i++){
          let tempSplit = mergedStoreEnsembleLoc[i].split(/([0-9]+)/)
          if (tempSplit[0] == 'KNN' || tempSplit[0] == 'KNNC' || tempSplit[0] == 'KNNM' || tempSplit[0] == 'KNNCC' || tempSplit[0] == 'KNNCM' || tempSplit[0] == 'KNNMC' || tempSplit[0] == 'KNNMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"KNN",value:this.PerF[tempSplit[1]], size:5, sw:true})
            } else {
              data.push({Algorithm:"KNN",value:this.PerF[tempSplit[1]], size:5, sw:false})
            }
          }
          else if (tempSplit[0] == 'LR' || tempSplit[0] == 'LRC' || tempSplit[0] == 'LRM' || tempSplit[0] == 'LRCC' || tempSplit[0] == 'LRCM' || tempSplit[0] == 'LRMC' || tempSplit[0] == 'LRMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"LR",value:this.PerF[tempSplit[1]], size:5, sw:true})
            } else {
              data.push({Algorithm:"LR",value:this.PerF[tempSplit[1]], size:5, sw:false})
            }
          }
          else if (tempSplit[0] == 'MLP' || tempSplit[0] == 'MLPC' || tempSplit[0] == 'MLPM' || tempSplit[0] == 'MLPCC' || tempSplit[0] == 'MLPCM' || tempSplit[0] == 'MLPMC' || tempSplit[0] == 'MLPMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"MLP",value:this.PerF[tempSplit[1]], size:5, sw:true})
            } else {
              data.push({Algorithm:"MLP",value:this.PerF[tempSplit[1]], size:5, sw:false})
            }
          }
          else if (tempSplit[0] == 'RF' || tempSplit[0] == 'RFC' || tempSplit[0] == 'RFM' || tempSplit[0] == 'RFCC' || tempSplit[0] == 'RFCM' || tempSplit[0] == 'RFMC' || tempSplit[0] == 'RFMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"RF",value:this.PerF[tempSplit[1]], size:5, sw:true})
            } else {
              data.push({Algorithm:"RF",value:this.PerF[tempSplit[1]], size:5, sw:false})
            }
          }
          else {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"GradB",value:this.PerF[tempSplit[1]], size:5, sw:true})
            } else {
              data.push({Algorithm:"GradB",value:this.PerF[tempSplit[1]], size:5, sw:false})
            }
          }
        }   
      }

        var widthChr = 589;
        var heightChr = 280;

        let svgAlg = d3v5
          .select("#MainPlot")
          .append("svg")
          .attr("height", heightChr)
          .attr("width", widthChr);

        let sectors = Array.from(new Set(data.map((d) => d.Algorithm)));
        let xCoords = sectors.map((d, i) => 95 + i * 108);
        let xScale = d3v5.scaleOrdinal().domain(sectors).range(xCoords);

        let yScale = d3v5
          .scaleLinear()
          .domain(d3v5.extent(data.map((d) => d.value)))
          .range([heightChr-30, 30]);

        var colorsF = d3.scale.ordinal().range(['#ff7f00','#fdbf6f','#fb9a99','#b15928','#a6cee3'])

        svgAlg
          .selectAll(".circ")
          .data(data)
          .enter()
          .append("circle")
          .attr("class", "circ")
          .attr("stroke", "black")
          .attr("fill", function (d) {
            if (d.sw) { return "#000000" } 
            else { return colorsF(d.Algorithm) }
          })
          .attr("r", (d) => d.size)
          .attr("cx", (d) => xScale(d.Algorithm))
          .attr("cy", function (d) {
            originalPositions.push(yScale(d.value))
            return yScale(d.value)
          });

        let simulation = d3v5
          .forceSimulation(data)
          .force(
            "x",
            d3v5
              .forceX((d) => {
                return xScale(d.Algorithm);
              })
              .strength(1)
          )
          .force(
            "y",
            d3v5
              .forceY(function (d) {
                return yScale(d.value);
              })
              .strength(1)
          )
          .force(
            "collide",
            d3v5.forceCollide((d) => {
              return d.size;
            })
          )
          .alphaDecay(0)
          .alpha(1)
          //.on("tick", tick);

        function tick() {
          d3v5.selectAll(".circ")
            .attr("cx", (d) => {
              return d.x;
            })
            .attr("cy", (d) => d.y);
        }

        simulation.alphaDecay(0.1);
        setTimeout(function () {
          tick();
          d3v5.selectAll(".circ").attr("cy", function (d) {
            belongs.push(d.Algorithm)
            newPositions.push(d.y)
            return d.y
          })
          for(var i = 0;i<=originalPositions.length-1;i++)
            difference.push(Math.ceil(Math.abs(newPositions[i] - originalPositions[i])));
          maximumDiff = Math.max(difference)
          minimumDiff = Math.min(difference)
          
          var localSum = new Array(5).fill(0);
          var localCount = new Array(5).fill(0);
          for (let j = 0; j<belongs.length; j++) {
            if(belongs[j] == "KNN") {
              localSum[0] = localSum[0] + difference[j]
              localCount[0]++
            }
            else if (belongs[j] == "LR") {
              localSum[1] = localSum[1] + difference[j]
              localCount[1]++
            } else if (belongs[j] == "MLP") {
              localSum[2] = localSum[2] + difference[j]
              localCount[2]++
            } else if (belongs[j] == "RF") {
              localSum[3] = localSum[3] + difference[j]
              localCount[3]++
            } else {
              localSum[4] = localSum[4] + difference[j]
              localCount[4]++
            }
          }
          for (let k = 0; k<localSum.length; k++) {
            average[k] = localSum[k]/localCount[k]
          }

          var dataPlot = [
            {
              x: ['KNN', 'LR', 'MLP', 'RF', 'GradB'],
              y: average,
              marker:{
                color: ['rgb(255,127,0)', 'rgb(253,191,111)', 'rgb(251,154,153)', 'rgb(177,89,40)', 'rgb(166,206,227)']
              },
              type: 'bar'
            }
          ];
          
          var layout = {
            xaxis: {
              visible: false
            },
            yaxis: {
              nticks: 2,
              range: [minimumDiff, maximumDiff],
              title: {
                text: 'Dev. (px)',
              }
            },
            title: { 
              text: 'Visualization Uncertainty (Mean Deviation in Pixels)', 
            },
            width: 589,
            height: 50,
            showlegend: false,
            bargap :0.60,
            margin: {
              l: 40,
              r: 0,
              b: 5,
              t: 20,
              pad: 0
            },
          };

          var config = {'displayModeBar': false}

          Plotly.newPlot('uncertainty', dataPlot, layout, config);

        }, 6000);

        chart1 = makeDistroChart({
          data:data,
          xName:'Algorithm',
          yName:'value',
          axisLabels: {xAxis: 'Algorithhm', yAxis: '# Ov. Performance (%) #'},
          selector:"#Bees",
          constrainExtremes:true});
          
        //chart1.renderDataPlots({showPlot:true,plotType:'beeswarm',showBeanLines:false, colors:null});


      
      


      /**
       * Creates a box plot, violin plot, and or notched box plot
       * @param settings Configuration options for the base plot
       * @param settings.data The data for the plot
       * @param settings.xName The name of the column that should be used for the x groups
       * @param settings.yName The name of the column used for the y values
       * @param {string} settings.selector The selector string for the main chart div
       * @param [settings.axisLabels={}] Defaults to the xName and yName
       * @param [settings.yTicks = 1] 1 = default ticks. 2 =  double, 0.5 = half
       * @param [settings.scale='linear'] 'linear' or 'log' - y scale of the chart
       * @param [settings.chartSize={width:800, height:400}] The height and width of the chart itself (doesn't include the container)
       * @param [settings.margin={top: 15, right: 60, bottom: 40, left: 50}] The margins around the chart (inside the main div)
       * @param [settings.constrainExtremes=false] Should the y scale include outliers?
       * @returns {object} chart A chart object
       */
      function makeDistroChart(settings) {

          var chart = {};

          // Defaults
          chart.settings = {
              data: null,
              xName: null,
              yName: null,
              selector: null,
              axisLables: null,
              yTicks: 0.25,
              scale: 'linear',
              chartSize: {width: 825, height: 354},
              margin: {top: 15, right: 45, bottom: 75, left: 40},
              constrainExtremes: false,
              color: ['#ff7f00','#fdbf6f','#fb9a99','#b15928','#a6cee3']
          };
          for (var setting in settings) {
              chart.settings[setting] = settings[setting]
          }


          function formatAsFloat(d) {
              if (d % 1 !== 0) {
                  return d3.format(".2f")(d);
              } else {
                  return d3.format(".0f")(d);
              }
          }

          function logFormatNumber(d) {
              var x = Math.log(d) / Math.log(10) + 1e-6;
              return Math.abs(x - Math.floor(x)) < 0.6 ? formatAsFloat(d) : "";
          }

          chart.yFormatter = formatAsFloat;

          chart.data = chart.settings.data;

          chart.groupObjs = {}; //The data organized by grouping and sorted as well as any metadata for the groups
          chart.objs = {mainDiv: null, chartDiv: null, g: null, xAxis: null, yAxis: null};
          chart.colorFunct = null;

          /**
           * Takes an array, function, or object mapping and created a color function from it
           * @param {function|[]|object} colorOptions
           * @returns {function} Function to be used to determine chart colors
           */
          function getColorFunct(colorOptions) {
              if (typeof colorOptions == 'function') {
                  return colorOptions
              } else if (Array.isArray(colorOptions)) {
                  //  If an array is provided, map it to the domain
                  var colorMap = {}, cColor = 0;
                  for (var cName in chart.groupObjs) {
                      colorMap[cName] = colorOptions[cColor];
                      cColor = (cColor + 1) % colorOptions.length;
                  }
                  return function (group) {
                      return colorMap[group];
                  }
              } else if (typeof colorOptions == 'object') {
                  // if an object is provided, assume it maps to  the colors
                  return function (group) {
                      return colorOptions[group];
                  }
              } else {
                  return d3.scale.ordinal().range(['#ff7f00','#fdbf6f','#fb9a99','#b15928','#a6cee3'])

              }
          }

          /**
           * Takes a percentage as returns the values that correspond to that percentage of the group range witdh
           * @param objWidth Percentage of range band
           * @param gName The bin name to use to get the x shift
           * @returns {{left: null, right: null, middle: null}}
           */
          function getObjWidth(objWidth, gName) {
              var objSize = {left: null, right: null, middle: null};
              var width = chart.xScale.rangeBand() * (objWidth / 100);
              var padding = (chart.xScale.rangeBand() - width) / 2;
              var gShift = chart.xScale(gName);
              objSize.middle = chart.xScale.rangeBand() / 2 + gShift;
              objSize.left = padding + gShift;
              objSize.right = objSize.left + width;
              return objSize;
          }

          /**
           * Adds jitter to the  scatter point plot
           * @param doJitter true or false, add jitter to the point
           * @param width percent of the range band to cover with the jitter
           * @returns {number}
           */
          function addJitter(doJitter, width) {
              if (doJitter !== true || width == 0) {
                  return 0
              }
              return Math.floor(Math.random() * width) - width / 2;
          }

          function shallowCopy(oldObj) {
              var newObj = {};
              for (var i in oldObj) {
                  if (oldObj.hasOwnProperty(i)) {
                      newObj[i] = oldObj[i];
                  }
              }
              return newObj;
          }

          /**
           * Closure that creates the tooltip hover function
           * @param groupName Name of the x group
           * @param metrics Object to use to get values for the group
           * @returns {Function} A function that provides the values for the tooltip
           */
          function tooltipHover(groupName, metrics) {
              var tooltipString = "Group: " + groupName;
              tooltipString += "<br\>Max: " + formatAsFloat(metrics.max, 0.1);
              tooltipString += "<br\>Q3: " + formatAsFloat(metrics.quartile3);
              tooltipString += "<br\>Median: " + formatAsFloat(metrics.median);
              tooltipString += "<br\>Q1: " + formatAsFloat(metrics.quartile1);
              tooltipString += "<br\>Min: " + formatAsFloat(metrics.min);
              return function () {
                  chart.objs.tooltip.transition().duration(200).style("opacity", 0.9);
                  chart.objs.tooltip.html(tooltipString)
              };
          }

          /**
           * Parse the data and calculates base values for the plots
           */
          !function prepareData() {
              function calcMetrics(values) {

                  var metrics = { //These are the original non�scaled values
                      max: null,
                      upperOuterFence: null,
                      upperInnerFence: null,
                      quartile3: null,
                      median: null,
                      mean: null,
                      iqr: null,
                      quartile1: null,
                      lowerInnerFence: null,
                      lowerOuterFence: null,
                      min: null
                  };

                  metrics.min = d3.min(values);
                  metrics.quartile1 = d3.quantile(values, 0.25);
                  metrics.median = d3.median(values);
                  metrics.mean = d3.mean(values);
                  metrics.quartile3 = d3.quantile(values, 0.75);
                  metrics.max = d3.max(values);
                  metrics.iqr = metrics.quartile3 - metrics.quartile1;

                  //The inner fences are the closest value to the IQR without going past it (assumes sorted lists)
                  var LIF = metrics.quartile1 - (1.5 * metrics.iqr);
                  var UIF = metrics.quartile3 + (1.5 * metrics.iqr);
                  for (var i = 0; i <= values.length; i++) {
                      if (values[i] < LIF) {
                          continue;
                      }
                      if (!metrics.lowerInnerFence && values[i] >= LIF) {
                          metrics.lowerInnerFence = values[i];
                          continue;
                      }
                      if (values[i] > UIF) {
                          metrics.upperInnerFence = values[i - 1];
                          break;
                      }
                  }


                  metrics.lowerOuterFence = metrics.quartile1 - (3 * metrics.iqr);
                  metrics.upperOuterFence = metrics.quartile3 + (3 * metrics.iqr);
                  if (!metrics.lowerInnerFence) {
                      metrics.lowerInnerFence = metrics.min;
                  }
                  if (!metrics.upperInnerFence) {
                      metrics.upperInnerFence = metrics.max;
                  }
                  return metrics
              }

              var current_x = null;
              var current_y = null;
              var current_row;

              // Group the values
              for (current_row = 0; current_row < chart.data.length; current_row++) {
                  current_x = chart.data[current_row][chart.settings.xName];
                  current_y = chart.data[current_row][chart.settings.yName];

                  if (chart.groupObjs.hasOwnProperty(current_x)) {
                      chart.groupObjs[current_x].values.push(current_y);
                  } else {
                      chart.groupObjs[current_x] = {};
                      chart.groupObjs[current_x].values = [current_y];
                  }
              }

              for (var cName in chart.groupObjs) {
                  chart.groupObjs[cName].values.sort(d3.ascending);
                  chart.groupObjs[cName].metrics = {};
                  chart.groupObjs[cName].metrics = calcMetrics(chart.groupObjs[cName].values);

              }
          }();

          /**
           * Prepare the chart settings and chart div and svg
           */
          !function prepareSettings() {
              //Set base settings
              chart.margin = chart.settings.margin;
              chart.divWidth = chart.settings.chartSize.width;
              chart.divHeight = chart.settings.chartSize.height;
              chart.width = chart.divWidth - chart.margin.left - chart.margin.right;
              chart.height = chart.divHeight - chart.margin.top - chart.margin.bottom;

              if (chart.settings.axisLabels) {
                  chart.xAxisLable = chart.settings.axisLabels.xAxis;
                  chart.yAxisLable = chart.settings.axisLabels.yAxis;
              } else {
                  chart.xAxisLable = chart.settings.xName;
                  chart.yAxisLable = chart.settings.yName;
              }

              if (chart.settings.scale === 'log') {
                  chart.yScale = d3.scale.log();
                  chart.yFormatter = logFormatNumber;
              } else {
                  chart.yScale = d3.scale.linear();
              }

              if (chart.settings.constrainExtremes === true) {
                  var fences = [];
                  for (var cName in chart.groupObjs) {
                      fences.push(chart.groupObjs[cName].metrics.lowerInnerFence);
                      fences.push(chart.groupObjs[cName].metrics.upperInnerFence);
                  }
                  chart.range = d3.extent(fences);

              } else {
                  chart.range = d3.extent(chart.data, function (d) {return d[chart.settings.yName];});
              }

              chart.colorFunct = getColorFunct(chart.settings.colors);

              // Build Scale functions
              chart.yScale.range([chart.height, 0]).domain(chart.range).nice().clamp(true);
              chart.xScale = d3.scale.ordinal().domain(Object.keys(chart.groupObjs)).rangeBands([0, chart.width]);

              //Build Axes Functions
              chart.objs.yAxis = d3.svg.axis()
                  .scale(chart.yScale)
                  .orient("left")
                  .tickFormat(chart.yFormatter)
                  .outerTickSize(0)
                  .innerTickSize(-chart.width + (chart.margin.right + chart.margin.left));
              chart.objs.yAxis.ticks(chart.objs.yAxis.ticks()*chart.settings.yTicks);
              chart.objs.xAxis = d3.svg.axis().scale(chart.xScale).orient("bottom").tickSize(5);
          }();

          /**
           * Updates the chart based on the current settings and window size
           * @returns {*}
           */
          chart.update = function () {
              // Update chart size based on view port size
              chart.width = parseInt(chart.objs.chartDiv.style("width"), 10) - (chart.margin.left + chart.margin.right) + 45; // chart width and height
              chart.height = parseInt(chart.objs.chartDiv.style("height"), 10) - (chart.margin.top + chart.margin.bottom) + 60;

              // Update scale functions
              chart.xScale.rangeBands([0, chart.width]);
              chart.yScale.range([chart.height, 0]);

              // Update the yDomain if the Violin plot clamp is set to -1 meaning it will extend the violins to make nice points
              if (chart.violinPlots && chart.violinPlots.options.show == true && chart.violinPlots.options._yDomainVP != null) {
                  chart.yScale.domain(chart.violinPlots.options._yDomainVP).nice().clamp(true);
              } else {
                  chart.yScale.domain(chart.range).nice().clamp(true);
              }

              //Update axes
              chart.objs.g.select('.x.axis').attr("transform", "translate(0," + chart.height + ")").call(chart.objs.xAxis)
                  .selectAll("text")
                  .attr("y", 10)
                  .attr("x", 15)
                  .attr("transform", "rotate(0)")
                  .style("text-anchor", "end");
              chart.objs.g.select('.x.axis .label').attr("x", chart.width / 2);
              chart.objs.g.select('.y.axis').call(chart.objs.yAxis.innerTickSize(-chart.width));
              chart.objs.g.select('.y.axis .label').attr("x", -chart.height / 2);
              chart.objs.chartDiv.select('svg').attr("width", chart.width + (chart.margin.left + chart.margin.right)).attr("height", chart.height + (chart.margin.top + chart.margin.bottom));

              return chart;
          };

          /**
           * Prepare the chart html elements
           */
          !function prepareChart() {
              // Build main div and chart div
              chart.objs.mainDiv = d3.select(chart.settings.selector)
                  .style("max-width", chart.divWidth + "px");
              // Add all the divs to make it centered and responsive
              chart.objs.mainDiv.append("div")
                  .attr("class", "inner-wrapper")
                  .append("div").attr("class", "outer-box")
                  .append("div").attr("class", "inner-box");
              // Capture the inner div for the chart (where the chart actually is)
              chart.selector = chart.settings.selector + " .inner-box";
              chart.objs.chartDiv = d3.select(chart.selector);
              d3.select(window).on('resize.' + chart.selector, chart.update);

              // Create the svg
              chart.objs.g = chart.objs.chartDiv.append("svg")
                  .attr("class", "chart-area")
                  .attr("width", chart.width + (chart.margin.left + chart.margin.right))
                  .attr("height", chart.height + (chart.margin.top + chart.margin.bottom))
                  .append("g")
                  .attr("transform", "translate(" + chart.margin.left + "," + chart.margin.top + ")");

              // Create axes
              chart.objs.axes = chart.objs.g.append("g").attr("class", "axis");
              chart.objs.axes.append("g")
                  .attr("class", "x axis")
                  .attr("transform", "translate(0," + chart.height + ")")
                  .call(chart.objs.xAxis);
              chart.objs.axes.append("g")
                  .attr("class", "y axis")
                  .call(chart.objs.yAxis)
                  .append("text")
                  .attr("class", "label")
                  .attr("transform", "rotate(-90)")
                  .attr("y", -35)
                  .attr("x", -chart.height / 2)
                  .attr("dy", ".71em")
                  .style("text-anchor", "middle")
                  .text(chart.yAxisLable);

              // Create tooltip div
              chart.objs.tooltip = chart.objs.mainDiv.append('div').attr('class', 'tooltip');
              for (var cName in chart.groupObjs) {
                  chart.groupObjs[cName].g = chart.objs.g.append("g").attr("class", "group");
                  chart.groupObjs[cName].g.on("mouseover", function () {
                      chart.objs.tooltip
                          .style("display", null)
                          .style("left", (d3.event.pageX) + "px")
                          .style("top", (d3.event.pageY - 28) + "px");
                  }).on("mouseout", function () {
                      chart.objs.tooltip.style("display", "none");
                  }).on("mousemove", tooltipHover(cName, chart.groupObjs[cName].metrics))
              }
              chart.update();
          }();

          /**
          * Render a raw data in various forms
          * @param options
          * @param [options.show=true] Toggle the whole plot on and off
          * @param [options.showPlot=false] True or false, show points
          * @param [options.plotType='none'] Options: no scatter = (false or 'none'); scatter points= (true or [amount=% of width (default=10)]); beeswarm points = ('beeswarm')
          * @param [options.pointSize=6] Diameter of the circle in pizels (not the radius)
          * @param [options.showLines=['median']] Can equal any of the metrics lines
          * @param [options.showbeanLines=false] Options: no lines = false
          * @param [options.beanWidth=20] % width
          * @param [options.colors=chart default]
          * @returns {*} The chart object
          *
          */
          chart.renderDataPlots = function (options) {
              chart.dataPlots = {};


              //Defaults
              var defaultOptions = {
                  show: true,
                  showPlot: false,
                  plotType: 'none',
                  pointSize: 8,
                  showLines: false,//['median'],
                  showBeanLines: false,
                  beanWidth: 20,
                  colors: null
              };
              chart.dataPlots.options = shallowCopy(defaultOptions);
              for (var option in options) {
                  chart.dataPlots.options[option] = options[option]
              }
              var dOpts = chart.dataPlots.options;

              //Create notch objects
              for (var cName in chart.groupObjs) {
                  chart.groupObjs[cName].dataPlots = {};
                  chart.groupObjs[cName].dataPlots.objs = {};
              }
              // The lines don't fit into a group bucket so they live under the dataPlot object
              chart.dataPlots.objs = {};

              /**
              * Take updated options and redraw the data plots
              * @param updateOptions
              */
              chart.dataPlots.change = function (updateOptions) {
                  if (updateOptions) {
                      for (var key in updateOptions) {
                          dOpts[key] = updateOptions[key]
                      }
                  }

                  chart.dataPlots.objs.g.remove();
                  for (var cName in chart.groupObjs) {
                      chart.groupObjs[cName].dataPlots.objs.g.remove()
                  }
                  chart.dataPlots.preparePlots();
                  chart.dataPlots.update()
              };

              chart.dataPlots.reset = function () {
                  chart.dataPlots.change(defaultOptions)
              };
              chart.dataPlots.show = function (opts) {
                  if (opts !== undefined) {
                      opts.show = true;
                      if (opts.reset) {
                          chart.dataPlots.reset()
                      }
                  } else {
                      opts = {show: true};
                  }
                  chart.dataPlots.change(opts)
              };
              chart.dataPlots.hide = function (opts) {
                  if (opts !== undefined) {
                      opts.show = false;
                      if (opts.reset) {
                          chart.dataPlots.reset()
                      }
                  } else {
                      opts = {show: false};
                  }
                  chart.dataPlots.change(opts)
              };

              /**
              * Update the data plot obj values
              */
              chart.dataPlots.update = function () {
                  var cName, cGroup, cPlot;

                  // Metrics lines
                  if (chart.dataPlots.objs.g) {
                      var halfBand = chart.xScale.rangeBand() / 2; // find the middle of each band
                      for (var cMetric in chart.dataPlots.objs.lines) {
                          chart.dataPlots.objs.lines[cMetric].line
                              .x(function (d) {
                                  return chart.xScale(d.x) + halfBand
                              });
                          chart.dataPlots.objs.lines[cMetric].g
                              .datum(chart.dataPlots.objs.lines[cMetric].values)
                              .attr('d', chart.dataPlots.objs.lines[cMetric].line);
                      }
                  }


                  for (cName in chart.groupObjs) {
                    cGroup = chart.groupObjs[cName];
                    cPlot = cGroup.dataPlots;

                    if (cPlot.objs.points) {
                    if (dOpts.plotType == 'beeswarm') {
                        var swarmBounds = getObjWidth(100, cName);
                        var yPtScale = chart.yScale.copy()
                            .range([Math.floor(chart.yScale.range()[0] / dOpts.pointSize), 0])
                            //.interpolate(d3.interpolateRound)
                            .domain(chart.yScale.domain());
                        var maxWidth = Math.ceil(chart.xScale.rangeBand() / dOpts.pointSize);

                        var ptsObj = {};
                        var cYBucket = null;

                        // //  Bucket points
                        for (var pt = 0; pt < cGroup.values.length; pt++) {
                            cYBucket = yPtScale(cGroup.values[pt]);
                            if (ptsObj.hasOwnProperty(cYBucket) !== true) {
                                ptsObj[cYBucket] = [];
                            }
                            ptsObj[cYBucket].push(cPlot.objs.points.pts[pt]
                                .attr("cx", swarmBounds.middle)
                                .attr("cy", yPtScale(cGroup.values[pt]) * dOpts.pointSize));
                        }
                        // //  Plot buckets
                        var rightMax = Math.min(swarmBounds.right - dOpts.pointSize);
                        for (var row in ptsObj) {
                            var leftMin = swarmBounds.left + (Math.max((maxWidth - ptsObj[row].length) / 2, 0) * dOpts.pointSize);
                            var col = 0;
                            for (pt in ptsObj[row]) {
                                ptsObj[row][pt].attr("cx", Math.min(leftMin + col * dOpts.pointSize, rightMax) + dOpts.pointSize / 2);
                                col++
                            }
                        }


                    } 
                }
                  }
              };

              /**
              * Create the svg elements for the data plots
              */
              chart.dataPlots.preparePlots = function () {
                  var cName, cPlot;

                  if (dOpts && dOpts.colors) {
                      chart.dataPlots.colorFunct = getColorFunct(dOpts.colors);
                  } else {
                      chart.dataPlots.colorFunct = chart.colorFunct
                  }

                  if (dOpts.show == false) {
                      return
                  }

                  for (cName in chart.groupObjs) {

                      cPlot = chart.groupObjs[cName].dataPlots;
                      cPlot.objs.g = chart.groupObjs[cName].g.append("g").attr("class", "data-plot");

                      // Points Plot
                      if (dOpts.showPlot) {
                          cPlot.objs.points = {g: null, pts: []};
                          cPlot.objs.points.g = cPlot.objs.g.append("g").attr("class", "points-plot");
                          for (var pt = 0; pt < chart.groupObjs[cName].values.length; pt++) {
                              cPlot.objs.points.pts.push(cPlot.objs.points.g.append("circle")
                                  .attr("class", function () { return "CirclePoint" })
                                  .attr('r', function () {
                                    var dataLoc = data.filter( i => cName.includes( i.Algorithm ) );
                                    return dataLoc[pt].size; 
                                  }) // Options is diameter, r takes radius so divide by 2
                                  .style("fill", function () {
                                    var dataLoc = data.filter( i => cName.includes( i.Algorithm ) );
                                    if (dataLoc[pt].sw) { return "#000000" } 
                                    else { return chart.dataPlots.colorFunct(cName) }
                                  }));
                          }
                      }
  
                }  
              };

              chart.dataPlots.preparePlots();
 
              d3.select(window).on('resize.' + chart.selector + '.dataPlot', chart.dataPlots.update);
              chart.dataPlots.update();
              return chart;
          };

          return chart;
      }
    }
  },
  mounted () {
    EventBus.$on('SendSelectedPointsUpdateIndicator', data => { this.selectedSimple = data })
    EventBus.$on('SendSelectedPointsUpdateIndicatorCM', data => { this.selectedEnsem = data })
    EventBus.$on('SendSelectedPointsUpdateIndicator', this.BeesFun)
    EventBus.$on('SendSelectedPointsUpdateIndicatorCM', this.BeesFun)

    EventBus.$on('SendStoredIDsInitial', data => { this.storedCM = data })
    EventBus.$on('SendPerformanceInitialAlgs', data => {
    this.PerFCM = data}) 

    EventBus.$on('SendPerformance', data => {
    this.PerF = data})
    EventBus.$on('SendStoredEnsembleHist', data => { this.storedEnsem = data })

    EventBus.$on('callAlgorithhms', this.BeesFun)

    EventBus.$on('Responsive', data => {
    this.WH = data})
    EventBus.$on('ResponsiveandChange', data => {
    this.WH = data})

    // reset the views
    EventBus.$on('resetViews', this.reset)
  }
}

</script>

<style>
/*Primary Chart*/

.chart-wrapper .inner-wrapper {
    position: relative;
    padding-bottom: 50%; /*Overwritten by the JS*/
    width: 100%;
}
.chart-wrapper .outer-box {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
}
.chart-wrapper .inner-box {
    width: 100%;
    height: 100%;
}

.chart-wrapper text {
  font-family: sans-serif;
  font-size: 18.5px;
}

.chart-wrapper .axis path,
.chart-wrapper .axis line {
    fill: none;
    stroke: #888;
    stroke-width: 2px;
    shape-rendering: crispEdges;
}

.chart-wrapper .y.axis .tick line {
    stroke: lightgrey;
    opacity: 0.6;
    stroke-dasharray: 2,1;
    stroke-width: 1;
    shape-rendering: crispEdges;

}

.chart-wrapper .x.axis .domain {
  display: none;
}

.chart-wrapper div.tooltip {
    position: absolute;
    text-align: left;
    padding: 3px;
    font: 12px sans-serif;
    background: lightcyan;
    border: 0px;
    border-radius: 1px;
    pointer-events: none;
    opacity: 0.7;
}

.chart-wrapper .box-plot circle.median {
    /*the script makes the circles the same color as the box, you can override this in the js*/
    fill: white !important;
}

@media (max-width:500px){
    .chart-wrapper .box-plot circle {display: none;}
}

/*Violin Plot*/

.chart-wrapper .violin-plot .area {
    shape-rendering: geometricPrecision;
    opacity: 0.4;
}


.axis text {
  font-size: 16px !important;
}

/* Point Plots*/
.chart-wrapper .points-plot .CirclePoint {
    stroke: black;
    stroke-width: 1px;
}

.chart-wrapper .metrics-lines {
    stroke-width: 4px;
}

/* Non-Chart Styles for demo*/
.chart-options  {
    min-width: 200px;
    font-size: 18.5px;
    font-family: sans-serif;
}
.chart-options button {
    margin: 3px;
    padding: 3px;
    font-size: 18.5px;
}
.chart-options p {
    display: inline;
}
@media (max-width:500px){
    .chart-options p {display: block;}
}

#containerForAllAlg {
  height: 100px;
  position: relative;
}
#MainPlot {
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
}
#MainPlot {
  z-index: 10;
}

.gtitle {
  transform: translate(-351px, -93px) !important;
  font-size: 18.5px !important;
  font-family: sans-serif !important;
}

.g-ytitle {
  transform: translate(0px, 0px) !important;
}

.ytitle {
  transform: rotate(-90deg) translate(67px, -530px) !important;
  font-size: 16px !important;
  font-family: sans-serif !important;
}

</style>