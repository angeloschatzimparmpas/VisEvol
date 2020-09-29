<template>
  <div id="Bees" class="chart-wrapper" style="min-height: 357px;"></div>
</template>

<script>
import * as Plotly from 'plotly.js'
import { EventBus } from '../main.js'

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
      var chart1
      var data = []

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
              data.push({Algorithm:"KNN",value:this.PerF[i], size:5, sw:true})
            } else {
              data.push({Algorithm:"KNN",value:this.PerF[i], size:5, sw:false})
            }
          }
          else if (tempSplit[0] == 'LR' || tempSplit[0] == 'LRC' || tempSplit[0] == 'LRM' || tempSplit[0] == 'LRCC' || tempSplit[0] == 'LRCM' || tempSplit[0] == 'LRMC' || tempSplit[0] == 'LRMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"LR",value:this.PerF[i], size:5, sw:true})
            } else {
              data.push({Algorithm:"LR",value:this.PerF[i], size:5, sw:false})
            }
          }
          else if (tempSplit[0] == 'MLP' || tempSplit[0] == 'MLPC' || tempSplit[0] == 'MLPM' || tempSplit[0] == 'MLPCC' || tempSplit[0] == 'MLPCM' || tempSplit[0] == 'MLPMC' || tempSplit[0] == 'MLPMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"MLP",value:this.PerF[i], size:5, sw:true})
            } else {
              data.push({Algorithm:"MLP",value:this.PerF[i], size:5, sw:false})
            }
          }
          else if (tempSplit[0] == 'RF' || tempSplit[0] == 'RFC' || tempSplit[0] == 'RFM' || tempSplit[0] == 'RFCC' || tempSplit[0] == 'RFCM' || tempSplit[0] == 'RFMC' || tempSplit[0] == 'RFMM') {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"RF",value:this.PerF[i], size:5, sw:true})
            } else {
              data.push({Algorithm:"RF",value:this.PerF[i], size:5, sw:false})
            }
          }
          else {
            if (this.selectedEnsem.includes(mergedStoreEnsembleLoc[i])) {
              data.push({Algorithm:"GradB",value:this.PerF[i], size:4, sw:true})
            } else {
              data.push({Algorithm:"GradB",value:this.PerF[i], size:4, sw:false})
            }
          }
        }   
      }

      chart1 = makeDistroChart({
            data:data,
            xName:'Algorithm',
            yName:'value',
            axisLabels: {xAxis: 'Algorithhm', yAxis: '# Performance (%) #'},
            selector:"#Bees",
            constrainExtremes:true});
        chart1.renderDataPlots({showPlot:true,plotType:'beeswarm',showBeanLines:false, colors:null});
        chart1.renderNotchBoxes({showNotchBox:false});
        chart1.renderViolinPlot({showViolinPlot:false});
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
              yTicks: 1,
              scale: 'linear',
              chartSize: {width: 825, height: 420},
              margin: {top: 15, right: 45, bottom: 25, left: 40},
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
           * Render a violin plot on the current chart
           * @param options
           * @param [options.showViolinPlot=true] True or False, show the violin plot
           * @param [options.resolution=100 default]
           * @param [options.bandwidth=10 default] May need higher bandwidth for larger data sets
           * @param [options.width=50] The max percent of the group rangeBand that the violin can be
           * @param [options.interpolation=''] How to render the violin
           * @param [options.clamp=0 default]
           *   0 = keep data within chart min and max, clamp once data = 0. May extend beyond data set min and max
           *   1 = clamp at min and max of data set. Possibly no tails
           *  -1 = extend chart axis to make room for data to interpolate to 0. May extend axis and data set min and max
           * @param [options.colors=chart default] The color mapping for the violin plot
           * @returns {*} The chart object
           */
          chart.renderViolinPlot = function (options) {
              chart.violinPlots = {};

              var defaultOptions = {
                  show: true,
                  showViolinPlot: true,
                  resolution: 100,
                  bandwidth: 20,
                  width: 50,
                  interpolation: 'cardinal',
                  clamp: 1,
                  colors: chart.colorFunct,
                  _yDomainVP: null // If the Violin plot is set to close all violin plots, it may need to extend the domain, that extended domain is stored here
              };
              chart.violinPlots.options = shallowCopy(defaultOptions);
              for (var option in options) {
                  chart.violinPlots.options[option] = options[option]
              }
              var vOpts = chart.violinPlots.options;

              // Create violin plot objects
              for (var cName in chart.groupObjs) {
                  chart.groupObjs[cName].violin = {};
                  chart.groupObjs[cName].violin.objs = {};
              }

              /**
               * Take a new set of options and redraw the violin
               * @param updateOptions
               */
              chart.violinPlots.change = function (updateOptions) {
                  if (updateOptions) {
                      for (var key in updateOptions) {
                          vOpts[key] = updateOptions[key]
                      }
                  }

                  for (var cName in chart.groupObjs) {
                      chart.groupObjs[cName].violin.objs.g.remove()
                  }

                  chart.violinPlots.prepareViolin();
                  chart.violinPlots.update();
              };

              chart.violinPlots.reset = function () {
                  chart.violinPlots.change(defaultOptions)
              };
              chart.violinPlots.show = function (opts) {
                  if (opts !== undefined) {
                      opts.show = true;
                      if (opts.reset) {
                          chart.violinPlots.reset()
                      }
                  } else {
                      opts = {show: true};
                  }
                  chart.violinPlots.change(opts);

              };

              chart.violinPlots.hide = function (opts) {
                  if (opts !== undefined) {
                      opts.show = false;
                      if (opts.reset) {
                          chart.violinPlots.reset()
                      }
                  } else {
                      opts = {show: false};
                  }
                  chart.violinPlots.change(opts);

              };

              /**
               * Update the violin obj values
               */
              chart.violinPlots.update = function () {
                  var cName, cViolinPlot;

                  for (cName in chart.groupObjs) {
                      cViolinPlot = chart.groupObjs[cName].violin;

                      // Build the violins sideways, so use the yScale for the xScale and make a new yScale
                      var xVScale = chart.yScale.copy();


                      // Create the Kernel Density Estimator Function
                      cViolinPlot.kde = kernelDensityEstimator(eKernel(vOpts.bandwidth), xVScale.ticks(vOpts.resolution));
                      cViolinPlot.kdedata = cViolinPlot.kde(chart.groupObjs[cName].values);

                      var interpolateMax = chart.groupObjs[cName].metrics.max,
                          interpolateMin = chart.groupObjs[cName].metrics.min;

                      if (vOpts.clamp == 0 || vOpts.clamp == -1) { //
                          // When clamp is 0, calculate the min and max that is needed to bring the violin plot to a point
                          // interpolateMax = the Minimum value greater than the max where y = 0
                          interpolateMax = d3.min(cViolinPlot.kdedata.filter(function (d) {
                              return (d.x > chart.groupObjs[cName].metrics.max && d.y == 0)
                          }), function (d) {
                              return d.x;
                          });
                          // interpolateMin = the Maximum value less than the min where y = 0
                          interpolateMin = d3.max(cViolinPlot.kdedata.filter(function (d) {
                              return (d.x < chart.groupObjs[cName].metrics.min && d.y == 0)
                          }), function (d) {
                              return d.x;
                          });
                          // If clamp is -1 we need to extend the axises so that the violins come to a point
                          if (vOpts.clamp == -1) {
                              kdeTester = eKernelTest(eKernel(vOpts.bandwidth), chart.groupObjs[cName].values);
                              if (!interpolateMax) {
                                  var interMaxY = kdeTester(chart.groupObjs[cName].metrics.max);
                                  var interMaxX = chart.groupObjs[cName].metrics.max;
                                  var count = 25; // Arbitrary limit to make sure we don't get an infinite loop
                                  while (count > 0 && interMaxY != 0) {
                                      interMaxY = kdeTester(interMaxX);
                                      interMaxX += 1;
                                      count -= 1;
                                  }
                                  interpolateMax = interMaxX;
                              }
                              if (!interpolateMin) {
                                  var interMinY = kdeTester(chart.groupObjs[cName].metrics.min);
                                  var interMinX = chart.groupObjs[cName].metrics.min;
                                  var count = 25;  // Arbitrary limit to make sure we don't get an infinite loop
                                  while (count > 0 && interMinY != 0) {
                                      interMinY = kdeTester(interMinX);
                                      interMinX -= 1;
                                      count -= 1;
                                  }
                                  interpolateMin = interMinX;
                              }

                          }
                          // Check to see if the new values are outside the existing chart range
                          //   If they are assign them to the master _yDomainVP
                          if (!vOpts._yDomainVP) vOpts._yDomainVP = chart.range.slice(0);
                          if (interpolateMin && interpolateMin < vOpts._yDomainVP[0]) {
                              vOpts._yDomainVP[0] = interpolateMin;
                          }
                          if (interpolateMax && interpolateMax > vOpts._yDomainVP[1]) {
                              vOpts._yDomainVP[1] = interpolateMax;
                          }


                      }


                      if (vOpts.showViolinPlot) {
                          chart.update();
                          xVScale = chart.yScale.copy();

                          // Need to recalculate the KDE because the xVScale changed
                          cViolinPlot.kde = kernelDensityEstimator(eKernel(vOpts.bandwidth), xVScale.ticks(vOpts.resolution));
                          cViolinPlot.kdedata = cViolinPlot.kde(chart.groupObjs[cName].values);
                      }

                      cViolinPlot.kdedata = cViolinPlot.kdedata
                          .filter(function (d) {
                              return (!interpolateMin || d.x >= interpolateMin)
                          })
                          .filter(function (d) {
                              return (!interpolateMax || d.x <= interpolateMax)
                          });
                  }
                  for (cName in chart.groupObjs) {
                      cViolinPlot = chart.groupObjs[cName].violin;

                      // Get the violin width
                      var objBounds = getObjWidth(vOpts.width, cName);
                      var width = (objBounds.right - objBounds.left) / 2;

                      var yVScale = d3.scale.linear()
                          .range([width, 0])
                          .domain([0, d3.max(cViolinPlot.kdedata, function (d) {return d.y;})])
                          .clamp(true);

                      var area = d3.svg.area()
                          .interpolate(vOpts.interpolation)
                          .x(function (d) {return xVScale(d.x);})
                          .y0(width)
                          .y1(function (d) {return yVScale(d.y);});

                      var line = d3.svg.line()
                          .interpolate(vOpts.interpolation)
                          .x(function (d) {return xVScale(d.x);})
                          .y(function (d) {return yVScale(d.y)});

                      if (cViolinPlot.objs.left.area) {
                          cViolinPlot.objs.left.area
                              .datum(cViolinPlot.kdedata)
                              .attr("d", area);
                          cViolinPlot.objs.left.line
                              .datum(cViolinPlot.kdedata)
                              .attr("d", line);

                          cViolinPlot.objs.right.area
                              .datum(cViolinPlot.kdedata)
                              .attr("d", area);
                          cViolinPlot.objs.right.line
                              .datum(cViolinPlot.kdedata)
                              .attr("d", line);
                      }

                      // Rotate the violins
                      cViolinPlot.objs.left.g.attr("transform", "rotate(90,0,0)   translate(0,-" + objBounds.left + ")  scale(1,-1)");
                      cViolinPlot.objs.right.g.attr("transform", "rotate(90,0,0)  translate(0,-" + objBounds.right + ")");
                  }
              };

              /**
              * Create the svg elements for the violin plot
              */
              chart.violinPlots.prepareViolin = function () {
                  var cName, cViolinPlot;

                  if (vOpts.colors) {
                      chart.violinPlots.color = getColorFunct(vOpts.colors);
                  } else {
                      chart.violinPlots.color = chart.colorFunct
                  }

                  if (vOpts.show == false) {return}

                  for (cName in chart.groupObjs) {
                      cViolinPlot = chart.groupObjs[cName].violin;

                      cViolinPlot.objs.g = chart.groupObjs[cName].g.append("g").attr("class", "violin-plot");
                      cViolinPlot.objs.left = {area: null, line: null, g: null};
                      cViolinPlot.objs.right = {area: null, line: null, g: null};

                      cViolinPlot.objs.left.g = cViolinPlot.objs.g.append("g");
                      cViolinPlot.objs.right.g = cViolinPlot.objs.g.append("g");

                      if (vOpts.showViolinPlot !== false) {
                          //Area
                          cViolinPlot.objs.left.area = cViolinPlot.objs.left.g.append("path")
                              .attr("class", "area")
                              .style("fill", chart.violinPlots.color(cName));
                          cViolinPlot.objs.right.area = cViolinPlot.objs.right.g.append("path")
                              .attr("class", "area")
                              .style("fill", chart.violinPlots.color(cName));

                          //Lines
                          cViolinPlot.objs.left.line = cViolinPlot.objs.left.g.append("path")
                              .attr("class", "line")
                              .attr("fill", 'none')
                              .style("stroke", chart.violinPlots.color(cName));
                          cViolinPlot.objs.right.line = cViolinPlot.objs.right.g.append("path")
                              .attr("class", "line")
                              .attr("fill", 'none')
                              .style("stroke", chart.violinPlots.color(cName));
                      }

                  }

              };


              function kernelDensityEstimator(kernel, x) {
                  return function (sample) {
                      return x.map(function (x) {
                          return {x:x, y:d3.mean(sample, function (v) {return kernel(x - v);})};
                      });
                  };
              }

              function eKernel(scale) {
                  return function (u) {
                      return Math.abs(u /= scale) <= 1 ? .75 * (1 - u * u) / scale : 0;
                  };
              }

              // Used to find the roots for adjusting violin axis
              // Given an array, find the value for a single point, even if it is not in the domain
              function eKernelTest(kernel, array) {
                  return function (testX) {
                      return d3.mean(array, function (v) {return kernel(testX - v);})
                  }
              }

              chart.violinPlots.prepareViolin();

              d3.select(window).on('resize.' + chart.selector + '.violinPlot', chart.violinPlots.update);
              chart.violinPlots.update();
              return chart;
          };

          /**
          * Render a notched box on the current chart
          * @param options
          * @param [options.show=true] Toggle the whole plot on and off
          * @param [options.showNotchBox=true] Show the notch box
          * @param [options.showLines=false] Show lines at the confidence intervals
          * @param [options.boxWidth=35] The width of the widest part of the box
          * @param [options.medianWidth=20] The width of the narrowist part of the box
          * @param [options.lineWidth=50] The width of the confidence interval lines
          * @param [options.notchStyle=null] null=traditional style, 'box' cuts out the whole notch in right angles
          * @param [options.colors=chart default] The color mapping for the notch boxes
          * @returns {*} The chart object
          */
          chart.renderNotchBoxes = function (options) {
              chart.notchBoxes = {};

              //Defaults
              var defaultOptions = {
                  show: true,
                  showNotchBox: true,
                  showLines: false,
                  boxWidth: 35,
                  medianWidth: 20,
                  lineWidth: 50,
                  notchStyle: null,
                  colors: null
              };
              chart.notchBoxes.options = shallowCopy(defaultOptions);
              for (var option in options) {
                  chart.notchBoxes.options[option] = options[option]
              }
              var nOpts = chart.notchBoxes.options;

              //Create notch objects
              for (var cName in chart.groupObjs) {
                  chart.groupObjs[cName].notchBox = {};
                  chart.groupObjs[cName].notchBox.objs = {};
              }

              /**
              * Makes the svg path string for a notched box
              * @param cNotch Current notch box object
              * @param notchBounds objBound object
              * @returns {string} A string in the proper format for a svg polygon
              */
              function makeNotchBox(cNotch, notchBounds) {
                  var scaledValues = [];
                  if (nOpts.notchStyle == 'box') {
                      scaledValues = [
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.quartile1)],
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.lowerNotch)],
                          [notchBounds.medianLeft, chart.yScale(cNotch.metrics.lowerNotch)],
                          [notchBounds.medianLeft, chart.yScale(cNotch.metrics.median)],
                          [notchBounds.medianLeft, chart.yScale(cNotch.metrics.upperNotch)],
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.upperNotch)],
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.quartile3)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.quartile3)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.upperNotch)],
                          [notchBounds.medianRight, chart.yScale(cNotch.metrics.upperNotch)],
                          [notchBounds.medianRight, chart.yScale(cNotch.metrics.median)],
                          [notchBounds.medianRight, chart.yScale(cNotch.metrics.lowerNotch)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.lowerNotch)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.quartile1)]
                      ];
                  } else {
                      scaledValues = [
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.quartile1)],
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.lowerNotch)],
                          [notchBounds.medianLeft, chart.yScale(cNotch.metrics.median)],
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.upperNotch)],
                          [notchBounds.boxLeft, chart.yScale(cNotch.metrics.quartile3)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.quartile3)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.upperNotch)],
                          [notchBounds.medianRight, chart.yScale(cNotch.metrics.median)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.lowerNotch)],
                          [notchBounds.boxRight, chart.yScale(cNotch.metrics.quartile1)]
                      ];
                  }
                  return scaledValues.map(function (d) {
                      return [d[0], d[1]].join(",");
                  }).join(" ");
              }

              /**
              * Calculate the confidence intervals
              */
              !function calcNotches() {
                  var cNotch, modifier;
                  for (var cName in chart.groupObjs) {
                      cNotch = chart.groupObjs[cName];
                      modifier = (1.57 * (cNotch.metrics.iqr / Math.sqrt(cNotch.values.length)));
                      cNotch.metrics.upperNotch = cNotch.metrics.median + modifier;
                      cNotch.metrics.lowerNotch = cNotch.metrics.median - modifier;
                  }
              }();

              /**
              * Take a new set of options and redraw the notch boxes
              * @param updateOptions
              */
              chart.notchBoxes.change = function (updateOptions) {
                  if (updateOptions) {
                      for (var key in updateOptions) {
                          nOpts[key] = updateOptions[key]
                      }
                  }

                  for (var cName in chart.groupObjs) {
                      chart.groupObjs[cName].notchBox.objs.g.remove()
                  }
                  chart.notchBoxes.prepareNotchBoxes();
                  chart.notchBoxes.update();
              };

              chart.notchBoxes.reset = function () {
                  chart.notchBoxes.change(defaultOptions)
              };
              chart.notchBoxes.show = function (opts) {
                  if (opts !== undefined) {
                      opts.show = true;
                      if (opts.reset) {
                          chart.notchBoxes.reset()
                      }
                  } else {
                      opts = {show: true};
                  }
                  chart.notchBoxes.change(opts)
              };
              chart.notchBoxes.hide = function (opts) {
                  if (opts !== undefined) {
                      opts.show = false;
                      if (opts.reset) {
                          chart.notchBoxes.reset()
                      }
                  } else {
                      opts = {show: false};
                  }
                  chart.notchBoxes.change(opts)
              };

              /**
              * Update the notch box obj values
              */
              chart.notchBoxes.update = function () {
                  var cName, cGroup;

                  for (cName in chart.groupObjs) {
                      cGroup = chart.groupObjs[cName];

                      // Get the box size
                      var boxBounds = getObjWidth(nOpts.boxWidth, cName);
                      var medianBounds = getObjWidth(nOpts.medianWidth, cName);

                      var notchBounds = {
                          boxLeft: boxBounds.left,
                          boxRight: boxBounds.right,
                          middle: boxBounds.middle,
                          medianLeft: medianBounds.left,
                          medianRight: medianBounds.right
                      };

                      // Notch Box
                      if (cGroup.notchBox.objs.notch) {
                          cGroup.notchBox.objs.notch
                              .attr("points", makeNotchBox(cGroup, notchBounds));
                      }
                      if (cGroup.notchBox.objs.upperLine) {
                          var lineBounds = null;
                          if (nOpts.lineWidth) {
                              lineBounds = getObjWidth(nOpts.lineWidth, cName)
                          } else {
                              lineBounds = objBounds
                          }

                          var confidenceLines = {
                              upper: chart.yScale(cGroup.metrics.upperNotch),
                              lower: chart.yScale(cGroup.metrics.lowerNotch)
                          };
                          cGroup.notchBox.objs.upperLine
                              .attr("x1", lineBounds.left)
                              .attr("x2", lineBounds.right)
                              .attr('y1', confidenceLines.upper)
                              .attr("y2", confidenceLines.upper);
                          cGroup.notchBox.objs.lowerLine
                              .attr("x1", lineBounds.left)
                              .attr("x2", lineBounds.right)
                              .attr('y1', confidenceLines.lower)
                              .attr("y2", confidenceLines.lower);
                      }
                  }
              };

              /**
              * Create the svg elements for the notch boxes
              */
              chart.notchBoxes.prepareNotchBoxes = function () {
                  var cName, cNotch;

                  if (nOpts && nOpts.colors) {
                      chart.notchBoxes.colorFunct = getColorFunct(nOpts.colors);
                  } else {
                      chart.notchBoxes.colorFunct = chart.colorFunct
                  }

                  if (nOpts.show == false) {
                      return
                  }

                  for (cName in chart.groupObjs) {
                      cNotch = chart.groupObjs[cName].notchBox;

                      cNotch.objs.g = chart.groupObjs[cName].g.append("g").attr("class", "notch-plot");

                      // Plot Box (default show)
                      if (nOpts.showNotchBox) {
                          cNotch.objs.notch = cNotch.objs.g.append("polygon")
                              .attr("class", "notch")
                              .style("fill", chart.notchBoxes.colorFunct(cName))
                              .style("stroke", chart.notchBoxes.colorFunct(cName));
                          //A stroke is added to the notch with the group color, it is
                          // hidden by default and can be shown through css with stroke-width
                      }

                      //Plot Confidence Lines (default hide)
                      if (nOpts.showLines) {
                          cNotch.objs.upperLine = cNotch.objs.g.append("line")
                              .attr("class", "upper confidence line")
                              .style("stroke", chart.notchBoxes.colorFunct(cName));

                          cNotch.objs.lowerLine = cNotch.objs.g.append("line")
                              .attr("class", "lower confidence line")
                              .style("stroke", chart.notchBoxes.colorFunct(cName));
                      }
                  }
              };
              chart.notchBoxes.prepareNotchBoxes();

              d3.select(window).on('resize.' + chart.selector + '.notchBox', chart.notchBoxes.update);
              chart.notchBoxes.update();
              return chart;
          };

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
                            .interpolate(d3.interpolateRound)
                            .domain(chart.yScale.domain());
                        var maxWidth = Math.ceil(chart.xScale.rangeBand() / dOpts.pointSize);
                        
                        var ptsObj = {};
                        var cYBucket = null;
                        //  Bucket points
                        for (var pt = 0; pt < cGroup.values.length; pt++) {
                            cYBucket = yPtScale(cGroup.values[pt]);
                            if (ptsObj.hasOwnProperty(cYBucket) !== true) {
                                ptsObj[cYBucket] = [];
                            }
                            ptsObj[cYBucket].push(cPlot.objs.points.pts[pt]
                                .attr("cx", swarmBounds.middle)
                                .attr("cy", yPtScale(cGroup.values[pt]) * dOpts.pointSize));
                        }
                        //  Plot buckets
                        var rightMax = Math.min(swarmBounds.right - dOpts.pointSize);
                        for (var row in ptsObj) {
                            var leftMin = swarmBounds.left + (Math.max((maxWidth - ptsObj[row].length) / 2, 0) * dOpts.pointSize);
                            var col = 0;
                            for (pt in ptsObj[row]) {
                                ptsObj[row][pt].attr("cx", Math.min(leftMin + col * dOpts.pointSize, rightMax) + dOpts.pointSize / 2);
                                col++
                            }
                        }
                    } else { // For scatter points and points with no scatter
                        var plotBounds = null,
                            scatterWidth = 0,
                            width = 0;
                        if (dOpts.plotType == 'scatter' || typeof dOpts.plotType == 'number') {
                            //Default scatter percentage is 20% of box width
                            scatterWidth = typeof dOpts.plotType == 'number' ? dOpts.plotType : 20;
                        }

                        plotBounds = getObjWidth(scatterWidth, cName);
                        width = plotBounds.right - plotBounds.left;

                        for (var pt = 0; pt < cGroup.values.length; pt++) {
                            cPlot.objs.points.pts[pt]
                                .attr("cx", plotBounds.middle + addJitter(true, width))
                                .attr("cy", chart.yScale(cGroup.values[pt]));
                        }
                    }
                }



                      if (cPlot.objs.bean) {
                          var beanBounds = getObjWidth(dOpts.beanWidth, cName);
                          for (var pt = 0; pt < cGroup.values.length; pt++) {
                              cPlot.objs.bean.lines[pt]
                                  .attr("x1", beanBounds.left)
                                  .attr("x2", beanBounds.right)
                                  .attr('y1', chart.yScale(cGroup.values[pt]))
                                  .attr("y2", chart.yScale(cGroup.values[pt]));
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

                  // Metrics lines
                  chart.dataPlots.objs.g = chart.objs.g.append("g").attr("class", "metrics-lines");
                  if (dOpts.showLines && dOpts.showLines.length > 0) {
                      chart.dataPlots.objs.lines = {};
                      var cMetric;
                      for (var line in dOpts.showLines) {
                          cMetric = dOpts.showLines[line];
                          chart.dataPlots.objs.lines[cMetric] = {};
                          chart.dataPlots.objs.lines[cMetric].values = [];
                          for (var cGroup in chart.groupObjs) {
                              chart.dataPlots.objs.lines[cMetric].values.push({
                                  x: cGroup,
                                  y: chart.groupObjs[cGroup].metrics[cMetric]
                              })
                          }
                          chart.dataPlots.objs.lines[cMetric].line = d3.svg.line()
                              .interpolate("cardinal")
                              .y(function (d) {
                                  return chart.yScale(d.y)
                              });
                          chart.dataPlots.objs.lines[cMetric].g = chart.dataPlots.objs.g.append("path")
                              .attr("class", "line " + cMetric)
                              .attr("data-metric", cMetric)
                              .style("fill", 'none')
                              .style("stroke", chart.colorFunct(cMetric));
                      }

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
                                  .attr("class", "point")
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
.chart-wrapper .points-plot .point {
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
</style>