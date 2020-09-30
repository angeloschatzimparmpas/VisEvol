<template>
<div>
  <div id="doubleBarChart"></div>
  <div id="legendFinal" class="text-center"></div>
</div>
</template>

<script>
import { EventBus } from '../main.js'

export default {
  name: 'VotingResults',
  data () {
    return {
      FinalResultsforPlot: [],
      Info: [],
      WH: [],
    }
  },
  methods: {
    reset () {
        var svg = d3.select("#doubleBarChart");
        svg.selectAll("*").remove();
        var svgLeg = d3.select("#legendFinal");
        svgLeg.selectAll("*").remove();
        this.FinalResultsforPlot = []
    },
    VotingResultsFun () {

      var svg = d3.select("#doubleBarChart");
      svg.selectAll("*").remove();
      
      var info = JSON.parse(this.Info[13])

      var dataLoc = []
      dataLoc[0] = Math.floor(JSON.parse(this.FinalResultsforPlot[0])*100)
      dataLoc[1] = Math.floor(JSON.parse(this.FinalResultsforPlot[1])*100)
      dataLoc[2] = Math.floor(JSON.parse(this.FinalResultsforPlot[2])*100)
      dataLoc[3] = Math.floor(JSON.parse(this.FinalResultsforPlot[3])*100)
      dataLoc[4] = Math.floor(JSON.parse(this.FinalResultsforPlot[4])*100)
      dataLoc[5] = Math.floor(JSON.parse(this.FinalResultsforPlot[5])*100)
      dataLoc[6] = Math.floor(JSON.parse(this.FinalResultsforPlot[6])*100)
      dataLoc[7] = Math.floor(JSON.parse(this.FinalResultsforPlot[7])*100)
      dataLoc[8] = Math.floor(JSON.parse(this.FinalResultsforPlot[8])*100)
      dataLoc[9] = Math.floor(JSON.parse(this.FinalResultsforPlot[9])*100)
      dataLoc[10] = Math.floor(JSON.parse(this.FinalResultsforPlot[10])*100)
      dataLoc[11] = Math.floor(JSON.parse(this.FinalResultsforPlot[11])*100)
      dataLoc[12] = Math.floor(JSON.parse(this.FinalResultsforPlot[12])*100)
      dataLoc[13] = Math.floor(JSON.parse(this.FinalResultsforPlot[13])*100)
      dataLoc[14] = Math.floor(JSON.parse(this.FinalResultsforPlot[14])*100)
      dataLoc[15] = Math.floor(JSON.parse(this.FinalResultsforPlot[15])*100)

      var data = [
        {'countries': 'Active Accuracy', 'infant.mortality': dataLoc[0], 'gdp': dataLoc[1], 'group': 1, 'color': '#1f78b4'},
        {'countries': 'Best Accuracy', 'infant.mortality': dataLoc[2], 'gdp': dataLoc[3], 'group': 1, 'color': '#e31a1c'},
        {'countries': 'Active Precision', 'infant.mortality': dataLoc[4], 'gdp': dataLoc[5], 'group': 2, 'color': '#1f78b4'},
        {'countries': 'Best Precision', 'infant.mortality': dataLoc[6], 'gdp': dataLoc[7], 'group': 2, 'color': '#e31a1c'},
        {'countries': 'Active Recall', 'infant.mortality': dataLoc[8], 'gdp': dataLoc[9], 'group': 3, 'color': '#1f78b4'},
        {'countries': 'Best Recall', 'infant.mortality': dataLoc[10], 'gdp': dataLoc[11], 'group': 3, 'color': '#e31a1c'},
        {'countries': 'Active F1-score', 'infant.mortality': dataLoc[12], 'gdp': dataLoc[13], 'group': 4, 'color': '#1f78b4'},
        {'countries': 'Best F1-score', 'infant.mortality': dataLoc[14], 'gdp': dataLoc[15], 'group': 4, 'color': '#e31a1c'}
      ]

      var labelArea = 160;
      var chart,
              width = 214,
              bar_height = 15,
              height = bar_height * 16;
      var rightOffset = width + labelArea;

      var lCol = "infant.mortality";
      var rCol = "gdp";
      var xFrom = d3.scale.linear()
        .domain([0,100])
        .range([0, width-60]);
      var xTo = d3.scale.linear()
        .domain([0,100])
        .range([0, width-40]);
      var y = d3.scale.ordinal()
        .rangeBands([20, height-58]);

      var chart = d3.select("#doubleBarChart")
        .append('svg')
        .attr('class', 'chart')
        .attr('width', labelArea + width + width)
        .attr('height', height);

      y.domain(data.map(function (d) {
        return d.countries;
      }));

      var yPosByIndex = function (d) {
        return (y(d.countries) + d.group*15);
      };

      chart.selectAll("rect.left")
              .data(data)
              .enter().append("rect")
              .attr("x", function (d) {
                  return width - xFrom(d[lCol]);
              })
              .attr("y", yPosByIndex)
              .attr("class", "left")
              .attr("width", function (d) {
                  return xFrom(d[lCol]);
              })
              .attr('fill', function (d) {
                return d.color;
              })
              .attr("height", y.rangeBand());
      chart.selectAll("text.leftscore")
              .data(data)
              .enter().append("text")
              .attr("x", function (d) {
                  return width - xFrom(d[lCol])-40;
              })
              .attr("y", function (d) {
                  return (y(d.countries) + y.rangeBand() / 2) + d.group*15;
              })
              .attr("dx", "20")
              .attr("dy", ".36em")
              .attr("text-anchor", "end")
              .attr('class', 'leftscore')
              .text(function(d){return d[lCol];});

      chart.selectAll("text.name")
              .data(data)
              .enter().append("text")
              .attr("x", (labelArea / 2) + width)
              .attr("y", function (d) {
                  return (y(d.countries) + y.rangeBand() / 2) + d.group*15;
              })
              .attr("dy", ".20em")
              .attr("text-anchor", "middle")
              .attr('class', 'name')
              .style("fill", function(d) {
                return "#000000"
                // if (d.countries.includes('Active')) {
                //   return "#1f78b4"
                // } else {
                //   return "#e31a1c"
                // }
              })
              .text(function(d){return d.countries;});

      chart.selectAll("rect.right")
              .data(data)
              .enter().append("rect")
              .attr("x", rightOffset)
              .attr("y", yPosByIndex)
              .attr("class", "right")
              .attr("width", function (d) {
                  return xTo(d[rCol]);
              })
              .attr('fill', function (d) {
                return d.color;
              })
              .attr("height", y.rangeBand());

      chart.selectAll("text.score")
              .data(data)
              .enter().append("text")
              .attr("x", function (d) {
                  return xTo(d[rCol]) + rightOffset+40;
              })
              .attr("y", function (d) {
                  return (y(d.countries) + y.rangeBand() / 2) + d.group*15;
              })
              .attr("dx", -5)
              .attr("dy", ".36em")
              .attr("text-anchor", "end")
              .attr('class', 'score')
              .text(function(d){return d[rCol];});

      chart.append("text").attr("x",width/3).attr("y", 20).attr("class","title").text(info[0]);
      chart.append("text").attr("x",width/3+rightOffset).attr("y", 20).attr("class","title").text(info[1]);
      chart.append("text").attr("x",width+labelArea/3).attr("y", 20).attr("class","title").text("Metrics");
    },
    legendColFinal () {        
    //==================================================
    var viewerWidth = this.WH[0]*2.5
    var viewerHeight = this.WH[1]*0.058
    var viewerPosTop = viewerHeight * 0.2;
    var cellSizeHeat = 10
    var legendElementWidth = cellSizeHeat * 3;

    // http://bl.ocks.org/mbostock/5577023
    var colors = ['#1f78b4','#fff','#fff','#fff','#e31a1c'];

    var svgLeg = d3.select("#legendFinal");
      svgLeg.selectAll("*").remove();
        
      var svgLeg = d3.select("#legendFinal").append("svg")
        .attr("width", viewerWidth/2)
        .attr("height", viewerHeight*1)
        .style("margin-top", "6px")

      var legend = svgLeg.append('g')
          .attr("class", "legend")
          .attr("transform", "translate(0,0)")
          .selectAll(".legendElement")
          .data([0, 1, 3, 4, 5])
          .enter().append("g")
          .attr("class", "legendElement");

      legend.append("svg:rect")
          .attr("x", function(d, i) {
              return (legendElementWidth * i) + 35;
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
            if (i == 0) {
              return "Active";
            } else if (i== 4) {
              return "Best";
            } else {
              return "";
            }

          })
          .attr("x", function(d, i) {
            if (i == 0) {
              return (legendElementWidth * i) + 32;
            } else {
              return (legendElementWidth * i) + 36; 
            }

          })
          .attr("y", (viewerPosTop + cellSizeHeat) + 10);
  },
  },
  mounted() {
    EventBus.$on('LegendPredictEnsem', this.legendColFinal)

    EventBus.$on('emittedEventCallingInfo', data => { this.Info = data }) 
    EventBus.$on('emittedEventCallingResultsPlot', data => { this.FinalResultsforPlot = data }) 
    EventBus.$on('emittedEventCallingResultsPlot', this.VotingResultsFun) 

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

.chart .right {
    stroke: white;
}

.chart .left {
    stroke: white;
}

.chart text {
    fill: black;
}

.chart text.name {
    fill: black;
}

.chart text.title {
    fill: black;
    font-weight: bold;
}
</style>