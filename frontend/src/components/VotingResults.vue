<template>
  <div id="doubleBarChart" style="min-height: 270px;"></div>
</template>

<script>
import { EventBus } from '../main.js'

export default {
  name: 'VotingResults',
  data () {
    return {
      FinalResultsforPlot: [],
      Info: [],
      NumberofExecutions: 0,
      scoresMean: [],
      scoresSTD: [],
      scoresPositive: [], 
      scoresNegative: [],
      scoresMean2: [],
      scoresSTD2: [],
      scoresPositive2: [], 
      scoresNegative2: [],
      scoresMean3: [],
      scoresSTD3: [],
      scoresPositive3: [], 
      scoresNegative3: [],
      scoresMean4: [],
      scoresSTD4: [],
      scoresPositive4: [], 
      scoresNegative4: [],
      Stack_scoresMean: [],
      Stack_scoresSTD: [],
      Stack_scoresPositive: [], 
      Stack_scoresNegative: [],
      Stack_scoresMean2: [],
      Stack_scoresSTD2: [],
      Stack_scoresPositive2: [], 
      Stack_scoresNegative2: [],
      Stack_scoresMean3: [],
      Stack_scoresSTD3: [],
      Stack_scoresPositive3: [], 
      Stack_scoresNegative3: [],
      Stack_scoresMean4: [],
      Stack_scoresSTD4: [],
      Stack_scoresPositive4: [], 
      Stack_scoresNegative4: [],
      xaxis: [],
      WH: [],
      firstTime: 0
    }
  },
  methods: {
    reset () {
        var svg = d3.select("#doubleBarChart");
        svg.selectAll("*").remove();
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
        {'countries': 'Active Accuracy', 'infant.mortality': dataLoc[0], 'gdp': dataLoc[1], 'group': 1, 'color': 'LIGHTSKYBLUE'},
        {'countries': 'Best Accuracy', 'infant.mortality': dataLoc[2], 'gdp': dataLoc[3], 'group': 1, 'color': 'LIGHTCORAL'},
        {'countries': 'Active Precision', 'infant.mortality': dataLoc[4], 'gdp': dataLoc[5], 'group': 2, 'color': 'LIGHTSKYBLUE'},
        {'countries': 'Best Precision', 'infant.mortality': dataLoc[6], 'gdp': dataLoc[7], 'group': 2, 'color': 'LIGHTCORAL'},
        {'countries': 'Active Recall', 'infant.mortality': dataLoc[8], 'gdp': dataLoc[9], 'group': 3, 'color': 'LIGHTSKYBLUE'},
        {'countries': 'Best Recall', 'infant.mortality': dataLoc[10], 'gdp': dataLoc[11], 'group': 3, 'color': 'LIGHTCORAL'},
        {'countries': 'Active F1-score', 'infant.mortality': dataLoc[12], 'gdp': dataLoc[13], 'group': 4, 'color': 'LIGHTSKYBLUE'},
        {'countries': 'Best F1-score', 'infant.mortality': dataLoc[14], 'gdp': dataLoc[15], 'group': 4, 'color': 'LIGHTCORAL'}
      ]

      var labelArea = 160;
      var chart,
              width = 214,
              bar_height = 15,
              height = bar_height * 18;
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
        .rangeBands([30, height-58]);

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

      chart.append("text").attr("x",width/3).attr("y", 15).attr("class","title").text(info[0]);
      chart.append("text").attr("x",width/3+rightOffset).attr("y", 15).attr("class","title").text(info[1]);
      chart.append("text").attr("x",width+labelArea/3).attr("y", 15).attr("class","title").text("Metrics");
    }
  },
  mounted() {
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