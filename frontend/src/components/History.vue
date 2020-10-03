<template>
  <div id="SankeyInter" style="min-height: 357px;"></div>
</template>

<script>
import * as Plotly from 'plotly.js'
import { EventBus } from '../main.js'
import { sankey  as d3Sankey } from 'd3-sankey'
import { format  as d3Format } from 'd3-format'
import * as d3Base from 'd3'
import $ from 'jquery'

// attach all d3 plugins to the d3 library
const d3v5 = Object.assign(d3Base)

export default {
  name: 'History',
  data () {
    return {
      WH: [],
      RandomSearLoc : 100,
      PerF: [],
      PerFCM: [],
      storedEnsem: [],
      storedCM: [],
      percentageOverall: [],
      values: [0,0,0,0,0,0,50,50,50,50,50,0,50,50,50,50,50,0],
      valuesStage2: [0,0,0,0,0,0,50,50,50,50,50,0,50,50,50,50,50,0,25,25,25,25,25,0,25,25,25,25,25,0,25,25,25,25,25,0,25,25,25,25,25,0],
      loop: 0,
      storePreviousPercentage: [],
      classesNumber: 9,
    }
  },
  methods: {
    reset () {
      var svg = d3.select("#SankeyInter");
      svg.selectAll("*").remove();
      var svgLeg = d3.select("#LegendSankey");
      svgLeg.selectAll("*").remove();
    },
    computePerformanceDiffS () {
      var colorsforScatterPlot = this.PerF
      var mergedStoreEnsembleLoc = [].concat.apply([], this.storedEnsem)
      var mergedStoreEnsembleLocFormatted = []
      for (let i = 0; i < mergedStoreEnsembleLoc.length; i++) {
        mergedStoreEnsembleLocFormatted.push(parseInt(mergedStoreEnsembleLoc[i].replace(/\D/g,'')))
      }
      
      colorsforScatterPlot = mergedStoreEnsembleLocFormatted.map((item) => colorsforScatterPlot[item])
      var max = Math.max.apply(Math, colorsforScatterPlot)
      var min = Math.min.apply(Math, colorsforScatterPlot)

      var countMax = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      var countMin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

      for (let i = 0; i < this.storedCM.length; i++) {
        let tempSplit = this.storedCM[i].split(/([0-9]+)/)

        if (tempSplit[0] == 'KNNCC') {
          if (this.PerFCM[i] > max) {
            countMax[0] = countMax[0] + 1
          } else if (this.PerFCM[i] < min) {
            countMin[0] = countMin[0] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'KNNCM') {
          if (this.PerFCM[i] > max) {
            countMax[1] = countMax[1] + 1
          } else if (this.PerFCM[i] < min) {
            countMin[1] = countMin[1] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'LRCC') {
          if (this.PerFCM[i] > max) {
            countMax[2] = countMax[2] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[2] = countMin[2] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'LRCM') {
          if (this.PerFCM[i] > max) {
            countMax[3] = countMax[3] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[3] = countMin[3] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'MLPCC') {
          if (this.PerFCM[i] > max) {
            countMax[4] = countMax[4] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[4] = countMin[4] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'MLPCM') {
          if (this.PerFCM[i] > max) {
            countMax[5] = countMax[5] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[5] = countMin[5] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'RFCC') {
          if (this.PerFCM[i] > max) {
            countMax[6] = countMax[6] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[6] = countMin[6] + 1
          }
        }
        else if (tempSplit[0] == 'RFCM') {
          if (this.PerFCM[i] > max) {
            countMax[7] = countMax[7] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[7] = countMin[7] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'GradBCC') {
          if (this.PerFCM[i] > max) {
            countMax[8] = countMax[8] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[8] = countMin[8] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'GradBCM') {
          if (this.PerFCM[i] > max) {
            countMax[9] = countMax[9] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[9] = countMin[9] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'KNNMC') {
          if (this.PerFCM[i] > max) {
            countMax[10] = countMax[10] + 1
          } else if (this.PerFCM[i] < min) {
            countMin[10] = countMin[10] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'KNNMM') {
          if (this.PerFCM[i] > max) {
            countMax[11] = countMax[11] + 1
          } else if (this.PerFCM[i] < min) {
            countMin[11] = countMin[11] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'LRMC') {
          if (this.PerFCM[i] > max) {
            countMax[12] = countMax[12] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[12] = countMin[12] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'LRMM') {
          if (this.PerFCM[i] > max) {
            countMax[13] = countMax[13] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[13] = countMin[13] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'MLPMC') {
          if (this.PerFCM[i] > max) {
            countMax[14] = countMax[14] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[14] = countMin[14] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'MLPMM') {
          if (this.PerFCM[i] > max) {
            countMax[15] = countMax[15] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[15] = countMin[15] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'RFMC') {
          if (this.PerFCM[i] > max) {
            countMax[16] = countMax[16] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[16] = countMin[16] + 1
          }
        }
        else if (tempSplit[0] == 'RFMM') {
          if (this.PerFCM[i] > max) {
            countMax[17] = countMax[17] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[17] = countMin[17] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'GradBMC') {
          if (this.PerFCM[i] > max) {
            countMax[18] = countMax[18] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[18] = countMin[18] + 1
          } else {
            continue
          }
        }
        else {
          if (this.PerFCM[i] > max) {
            countMax[19] = countMax[19] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[19] = countMin[19] + 1
          } else {
            continue
          }
        }
      }

      var percentage = []
      for (let j = 0; j < countMax.length; j++) {
        if (j >= 15) {
          if (countMax[j] == 0) {
            percentage.push((countMin[j]/2)*(-1)*100)
          } else {
            percentage.push(countMax[j]/2 * 100)
          }  
        } else if (j >= 10) {
          if (countMax[j] == 0) {
            percentage.push((countMin[j]/2)*(-1)*100)
          } else {
            percentage.push(countMax[j]/2 * 100)
          }  
        } else if (j >= 5) {
          if (countMax[j] == 0) {
            percentage.push((countMin[j]/2)*(-1)*100)
          } else {
            percentage.push(countMax[j]/2 * 100)
          }  
        } else {
          if (countMax[j] == 0) {
            percentage.push((countMin[j]/2)*(-1) * 100)
          } else {
            percentage.push(countMax[j]/2 * 100)
          }
        }
      }
//CORRECT
      // var percentage = []
      // for (let j = 0; j < countMax.length; j++) {
      //   if (j >= 15) {
      //     if (countMax[j] == 0) {
      //       percentage.push((countMin[j]/this.values[37-j])*(-1)*100)
      //     } else {
      //       percentage.push(countMax[j]/this.values[37-j] * 100)
      //     }  
      //   } else if (j >= 10) {
      //     if (countMax[j] == 0) {
      //       percentage.push((countMin[j]/this.values[38-j])*(-1)*100)
      //     } else {
      //       percentage.push(countMax[j]/this.values[38-j] * 100)
      //     }  
      //   } else if (j >= 5) {
      //     if (countMax[j] == 0) {
      //       percentage.push((countMin[j]/this.values[39-j])*(-1)*100)
      //     } else {
      //       percentage.push(countMax[j]/this.values[39-j] * 100)
      //     }  
      //   } else {
      //     if (countMax[j] == 0) {
      //       percentage.push((countMin[j]/this.values[40-j])*(-1) * 100)
      //     } else {
      //       percentage.push(countMax[j]/this.values[40-j] * 100)
      //     }
      //   }
      // }

      this.percentageOverall = percentage
      
    },
    SankeyViewStage3 () {
      var valuesLoc = this.valuesStage2
      var localStep = 2
      var numberofModels = 6
      var units = "Models";
      var initialModels = this.RandomSearLoc * 5
      //this is the svg canvas attributes: (not buidlign abything just seeting up varaibels)
      var margin = {top: 0, right: 40, bottom: 0, left: 40}, //comma is the equivalent of var : 
          width = 1250 - margin.left - margin.right,
          height = 350 - margin.top - margin.bottom;

      var svg = d3.select("#SankeyInter");
      svg.selectAll("*").remove();

      var formatNumber = d3Format(",.0f"),    // zero decimal places
        format = function(d) { return formatNumber(d) + " " + units; }
      var color = d3.scale.category20b()

      // append the svg canvas to the page
      var svg = d3.select("#SankeyInter").append("svg") //will select the id of cahrt from index.html ln:135 --> # looks for the id= from html
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g") //group everything on the vancas together.  will edit down on ln38 below
          .attr("transform", 
                "translate(" + margin.left + "," + margin.top + ") scale(1,-1) translate(" + 0 + "," + -height + ")");

      // Set the sankey diagram properties
      var sankey = d3Sankey() //calling the function
          .nodeWidth(25)
          .nodePadding(0)
          .size([width, height]);

      var path = sankey.link(); //sankey.link() is something happening in sankey.js 


      // load the data
      var graph = {
        "nodes":[
        {"name":"GradB","node":0,"month":"RandSear","color":"#a6cee3",valueN:this.RandomSearLoc},
        {"name":"RF","node":1,"month":"RandSear","color":"#b15928",valueN:this.RandomSearLoc},
        {"name":"MLP","node":2,"month":"RandSear","color":"#fb9a99",valueN:this.RandomSearLoc},
        {"name":"LR","node":3,"month":"RandSear","color":"#fdbf6f",valueN:this.RandomSearLoc},
        {"name":"KNN","node":4,"month":"RandSear","color":"#ff7f00",valueN:this.RandomSearLoc},
        {"name":"Random Search \u2014 S0","node":5,"month":"RandSear","color":"#ffffff",valueN:this.RandomSearLoc},
        {"name":"GradB","node":6,"month":"Crossover_S1","color":"#a6cee3",valueN:this.RandomSearLoc/2},
        {"name":"RF","node":7,"month":"Crossover_S1","color":"#b15928",valueN:this.RandomSearLoc/2},
        {"name":"MLP","node":8,"month":"Crossover_S1","color":"#fb9a99",valueN:this.RandomSearLoc/2},
        {"name":"LR","node":9,"month":"Crossover_S1","color":"#fdbf6f",valueN:this.RandomSearLoc/2},
        {"name":"KNN","node":10,"month":"Crossover_S1","color":"#ff7f00",valueN:this.RandomSearLoc/2},
        {"name":"(M) Mutate \u2014 S1","node":11,"month":"Crossover_S1","color":"#ffffff",valueN:this.RandomSearLoc/2},
        {"name":"GradB","node":12,"month":"Mutate_S1","color":"#a6cee3",valueN:this.RandomSearLoc/2},
        {"name":"RF","node":13,"month":"Mutate_S1","color":"#b15928",valueN:this.RandomSearLoc/2},
        {"name":"MLP","node":14,"month":"Mutate_S1","color":"#fb9a99",valueN:this.RandomSearLoc/2},
        {"name":"LR","node":15,"month":"Mutate_S1","color":"#fdbf6f",valueN:this.RandomSearLoc/2},
        {"name":"KNN","node":16,"month":"Mutate_S1","color":"#ff7f00",valueN:this.RandomSearLoc/2},
        {"name":"(C) Crossover \u2014 S1","node":17,"month":"Mutate_S1","color":"#ffffff",valueN:this.RandomSearLoc/2},
        {"name":"GradB","node":18,"month":"Crossover_S2","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":19,"month":"Crossover_S2","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":20,"month":"Crossover_S2","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":21,"month":"Crossover_S2","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":22,"month":"Crossover_S2","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Mutate \u2014 S2 (M)","node":23,"month":"Crossover_S2","color":"#ffffff",valueN:this.RandomSearLoc/4},
        {"name":"GradB","node":24,"month":"Mutate_S2","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":25,"month":"Mutate_S2","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":26,"month":"Mutate_S2","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":27,"month":"Mutate_S2","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":28,"month":"Mutate_S2","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Crossover \u2014 S2 (M)","node":29,"month":"Mutate_S2","color":"#ffffff",valueN:this.RandomSearLoc/4},
        {"name":"GradB","node":30,"month":"Crossover_S2_Prime","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":31,"month":"Crossover_S2_Prime","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":32,"month":"Crossover_S2_Prime","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":33,"month":"Crossover_S2_Prime","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":34,"month":"Crossover_S2_Prime","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Mutate \u2014 S2 (C)","node":35,"month":"Crossover_S2_Prime","color":"#ffffff",valueN:this.RandomSearLoc/4},
        {"name":"GradB","node":36,"month":"Mutate_S2_Prime","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":37,"month":"Mutate_S2_Prime","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":38,"month":"Mutate_S2_Prime","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":39,"month":"Mutate_S2_Prime","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":40,"month":"Mutate_S2_Prime","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Crossover \u2014 S2 (C)","node":41,"month":"Mutate_S2_Prime","color":"#ffffff",valueN:this.RandomSearLoc/4},
        ],

        "links":[
        {"source":5,"target":11,"value":0},
        {"source":5,"target":17,"value":0},
        {"source":0,"target":6,"value":valuesLoc[6]},
        {"source":0,"target":12,"value":valuesLoc[12]},
        {"source":1,"target":7,"value":valuesLoc[7]},
        {"source":1,"target":13,"value":valuesLoc[13]},
        {"source":2,"target":8,"value":valuesLoc[8]},
        {"source":2,"target":14,"value":valuesLoc[14]},
        {"source":3,"target":9,"value":valuesLoc[9]},
        {"source":3,"target":15,"value":valuesLoc[15]},
        {"source":4,"target":10,"value":valuesLoc[10]},
        {"source":4,"target":16,"value":valuesLoc[16]},
        {"source":11,"target":23,"value":0},
        {"source":11,"target":35,"value":0},
        {"source":6,"target":18,"value":valuesLoc[18]},
        {"source":6,"target":24,"value":valuesLoc[24]},
        {"source":7,"target":19,"value":valuesLoc[19]},
        {"source":7,"target":25,"value":valuesLoc[25]},
        {"source":8,"target":20,"value":valuesLoc[20]},
        {"source":8,"target":26,"value":valuesLoc[26]},
        {"source":9,"target":21,"value":valuesLoc[21]},
        {"source":9,"target":27,"value":valuesLoc[27]},
        {"source":10,"target":22,"value":valuesLoc[22]},
        {"source":10,"target":28,"value":valuesLoc[28]},
        {"source":17,"target":29,"value":0},
        {"source":17,"target":41,"value":0},
        {"source":12,"target":30,"value":valuesLoc[30]},
        {"source":12,"target":36,"value":valuesLoc[36]},
        {"source":13,"target":31,"value":valuesLoc[31]},
        {"source":13,"target":37,"value":valuesLoc[37]},
        {"source":14,"target":32,"value":valuesLoc[32]},
        {"source":14,"target":38,"value":valuesLoc[38]},
        {"source":15,"target":33,"value":valuesLoc[33]},
        {"source":15,"target":39,"value":valuesLoc[39]},
        {"source":16,"target":34,"value":valuesLoc[34]},
        {"source":16,"target":40,"value":valuesLoc[40]},
      ]}

        sankey.nodes(graph.nodes)
          .links(graph.links)
          .layout(0);
      var colorDiff
      colorDiff = d3v5.scaleSequential(d3v5.interpolatePRGn).domain([-100, 100])
      var percentage = this.percentageOverall

      var previousPercentage = this.storePreviousPercentage

      // add in the links
        var link = svg.append("g").selectAll(".link")
            .data(graph.links)
          .enter().append("path")
            .attr("class", "link")
            .attr("d", path) //d??? look it up later 
            .style("stroke",function(d){
              if(d.source.node == 5){
                return "transparent";
              }
              if(d.source.node == 11){
                return "transparent";
              }
              if(d.source.node == 17){
                return "transparent";
              } 
              if(d.target.node == 40){
                return colorDiff(percentage[0]);
              } else if(d.target.node == 39){
                return colorDiff(percentage[1]);
              } else if(d.target.node == 38){
                return colorDiff(percentage[2]);
              } else if(d.target.node == 37){
                return colorDiff(percentage[3]);
              } else if(d.target.node == 36){
                return colorDiff(percentage[4]);
              } else if(d.target.node == 34){
                return colorDiff(percentage[5]);
              } else if(d.target.node == 33){
                return colorDiff(percentage[6]);
              } else if(d.target.node == 32){
                return colorDiff(percentage[7]);
              } else if(d.target.node == 31){
                return colorDiff(percentage[8]);
              } else if(d.target.node == 30){
                return colorDiff(percentage[9]);
              } else if(d.target.node == 28){
                return colorDiff(percentage[10]);
              } else if(d.target.node == 27){
                return colorDiff(percentage[11]);
              } else if(d.target.node == 26){
                return colorDiff(percentage[12]);
              } else if(d.target.node == 25){
                return colorDiff(percentage[13]);
              } else if(d.target.node == 24){
                return colorDiff(percentage[14]);
              } else if(d.target.node == 22){
                return colorDiff(percentage[15]);
              } else if(d.target.node == 21){
                return colorDiff(percentage[16]);
              } else if(d.target.node == 20){
                return colorDiff(percentage[17]);
              } else if(d.target.node == 19){
                return colorDiff(percentage[18]);
              } else if(d.target.node == 18){
                return colorDiff(percentage[19]);
              } else if(d.target.node == 16){
                return colorDiff(previousPercentage[0]);
              } else if(d.target.node == 15){
                return colorDiff(previousPercentage[1]);
              } else if(d.target.node == 14){
                return colorDiff(previousPercentage[2]);
              } else if(d.target.node == 13){
                return colorDiff(previousPercentage[3]);
              } else if(d.target.node == 12){
                return colorDiff(previousPercentage[4]);
              } else if(d.target.node == 10){
                return colorDiff(previousPercentage[5]);
              } else if(d.target.node == 9){
                return colorDiff(previousPercentage[6]);
              } else if(d.target.node == 8){
                return colorDiff(previousPercentage[7]);
              } else if(d.target.node == 7){
                return colorDiff(previousPercentage[8]);
              } else if(d.target.node == 6){
                return colorDiff(previousPercentage[9]);
              } else {
                return "#808080"
              }
            }) 
            .style("stroke-width", function(d) { return Math.max(.5, d.dy); })   //setting the stroke length by the data . d.dy is defined in sankey.js
            .sort(function(a, b) { return b.dy - a.dy; })
            .on("mouseover",linkmouseover)
            .on("mouseout",linkmouseout);  

      // add the link titles
        link.append("svg:title") //this is the mouseover stuff title is an svg element you can use "svg:title" or just "title"
              .text(function(d) {
              return format(d.value); });

      // add in the nodes (creating the groups of the rectanlges)
        var node = svg.append("g").selectAll(".node") 
            .data(graph.nodes)
          .enter().append("g")
            .attr("class", "node")
            .attr("transform", function(d) { 
                return "translate(" + d.x + "," + d.y + ")";
            });
        
      // add the rectangles for the nodes
        node.append("rect")
            .attr("height", function(d) { return d.dy; })
            .attr("width", sankey.nodeWidth(  ))
            .style("fill", function(d) { return d.color; }) //matches name with the colors here! inside the replace is some sort of regex
            // .style("stroke",function(d) { return d3.rgb(d.color).darker(1); }) //line around the box formatting
            // .style("stroke-width",.5)
            .on("mouseover", nodemouseover)
            .on("mouseout", nodemouseout);

        // add in the title for the nodes
        node.append("text")
            .attr("x", -6)
            .attr("y", function(d) {
              if (d.node <= 5) {
                return d.dy - 81; 
              } else if (d.node <= 17) {
                return d.dy - 41; 
              } else {
                return d.dy - 21; 
              }
            })
            .attr("dy", ".35em")
            .attr("text-anchor", "end")
            .attr("transform", "scale(1,-1)")
            .text(function(d) { return d.name.replace(/-.*/, ""); })
            .style("font-weight", function(d) {
              if (d.node == 5 || d.node == 11 || d.node == 17 || d.node == 23 || d.node == 29 || d.node == 35 || d.node == 41) {
                return "bold";
              }
            })
            .style("font-size", function(d) {
              if (d.node > 17) {
                return "14.5px";
              }
            })
          .filter(function(d) { return d.x < width / 2; })//positioning left or right of node
            .attr("x", 6 + sankey.nodeWidth())
            .attr("text-anchor", "start");


      var status=null;
      function nodemouseover(d){

        d3.selectAll(".link")
            .attr("id", function(i){
              if (i.source.node == d.node || i.target.node == d.node){
                status="clicked";
              } else {
                status=null;
              }
              return status;
          });
          }

      function nodemouseout(d){
        d3.selectAll(".link")
            .attr("id", "unclicked");
          }
      function linkmouseover(d){
        d3.select(this)
            .attr("stroke-opacity",.8);
          }
      function linkmouseout(d){
        d3.select(this)
            .attr("stroke-opacity",.3);
          }

      //select all of our links and set a new stroke color on the conditioan that the value is =.01. 
      d3.selectAll(".link")
            .style("stroke-opacity", function(d){ 
                    if(d.value == 0.01) return 0;
                    });

      svg.append("g")
        .append("text")
        .text("Number of Models")
        .attr("font-family","sans-serif")
        .attr("font-size",18.5)
        .attr("fill","black")
        .attr("x",20)
        .attr("y",30)
        .attr("transform", 
                "translate(" + -45 + "," + 0 + ") scale(1,-1) translate(" + 0 + "," + -(height) + ") rotate(-90 150 150)");


    },
    computePerformanceDiff () {
      var colorsforScatterPlot = this.PerF

      var mergedStoreEnsembleLoc = [].concat.apply([], this.storedEnsem)
      var mergedStoreEnsembleLocFormatted = []
      for (let i = 0; i < mergedStoreEnsembleLoc.length; i++) {
        mergedStoreEnsembleLocFormatted.push(parseInt(mergedStoreEnsembleLoc[i].replace(/\D/g,'')))
      }
      
      colorsforScatterPlot = mergedStoreEnsembleLocFormatted.map((item) => colorsforScatterPlot[item])

      var max = Math.max.apply(Math, colorsforScatterPlot)
      var min = Math.min.apply(Math, colorsforScatterPlot)

      var countMax = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      var countMin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

      for (let i = 0; i < this.storedCM.length; i++) {
        let tempSplit = this.storedCM[i].split(/([0-9]+)/)

        if (tempSplit[0] == 'KNNC') {
          if (this.PerFCM[i] > max) {
            countMax[0] = countMax[0] + 1
          } else if (this.PerFCM[i] < min) {
            countMin[0] = countMin[0] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'KNNM') {
          if (this.PerFCM[i] > max) {
            countMax[1] = countMax[1] + 1
          } else if (this.PerFCM[i] < min) {
            countMin[1] = countMin[1] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'LRC') {
          if (this.PerFCM[i] > max) {
            countMax[2] = countMax[2] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[2] = countMin[2] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'LRM') {
          if (this.PerFCM[i] > max) {
            countMax[3] = countMax[3] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[3] = countMin[3] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'MLPC') {
          if (this.PerFCM[i] > max) {
            countMax[4] = countMax[4] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[4] = countMin[4] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'MLPM') {
          if (this.PerFCM[i] > max) {
            countMax[5] = countMax[5] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[5] = countMin[5] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'RFC') {
          if (this.PerFCM[i] > max) {
            countMax[6] = countMax[6] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[6] = countMin[6] + 1
          }
        }
        else if (tempSplit[0] == 'RFM') {
          if (this.PerFCM[i] > max) {
            countMax[7] = countMax[7] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[7] = countMin[7] + 1
          } else {
            continue
          }
        }
        else if (tempSplit[0] == 'GradBC') {
          if (this.PerFCM[i] > max) {
            countMax[8] = countMax[8] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[8] = countMin[8] + 1
          } else {
            continue
          }
        }
        else {
          if (this.PerFCM[i] > max) {
            countMax[9] = countMax[9] + 1
          }
          else if (this.PerFCM[i] < min) {
            countMin[9] = countMin[9] + 1
          } else {
            continue
          }
        }
      }

      var percentage = []
      for (let j = 0; j < countMax.length; j++) {
        if (j >= 5) {
          if (countMax[j] == 0) {
            percentage.push((countMin[j]/5)*(-1)*100)
          } else {
            percentage.push(countMax[j]/5 * 100)
          }  
        } else {
          if (countMax[j] == 0) {
            percentage.push((countMin[j]/5)*(-1) * 100)
          } else {
            percentage.push(countMax[j]/5 * 100)
          }
        }
      }
//CORRECT
      // var percentage = []
      // for (let j = 0; j < countMax.length; j++) {
      //   if (j >= 5) {
      //     if (countMax[j] == 0) {
      //       percentage.push((countMin[j]/this.values[15-j])*(-1)*100)
      //     } else {
      //       percentage.push(countMax[j]/this.values[15-j] * 100)
      //     }  
      //   } else {
      //     if (countMax[j] == 0) {
      //       percentage.push((countMin[j]/this.values[16-j])*(-1) * 100)
      //     } else {
      //       percentage.push(countMax[j]/this.values[16-j] * 100)
      //     }
      //   }
      // }

      this.percentageOverall = percentage
      
    },
    SankeyViewStage2 () {
      var valuesLoc = this.valuesStage2
      var localStep = 2
      var numberofModels = 6
      var units = "Models";
      var initialModels = this.RandomSearLoc * 5
      //this is the svg canvas attributes: (not buidlign abything just seeting up varaibels)
      var margin = {top: 0, right: 40, bottom: 0, left: 40}, //comma is the equivalent of var : 
          width = 1230 - margin.left - margin.right,
          height = 350 - margin.top - margin.bottom;

      var svg = d3.select("#SankeyInter");
      svg.selectAll("*").remove();

      var formatNumber = d3Format(",.0f"),    // zero decimal places
        format = function(d) { return formatNumber(d) + " " + units; }
      var color = d3.scale.category20b()

      var startingAxis = this.RandomSearLoc
      // var axisScale = d3.scale.linear()
      //                   .domain([startingAxis*6,0])
      //                   .range([0, height]);

      // //Create the Axis
      // var yAxis = d3.svg.axis()
      //               .scale(axisScale)
      //               .orient("left")
      //               .ticks(10);



      // var lossScale = d3.scale.linear()
      //                   .domain([.95,1,1.05])
      //                   .range(["red","black","green"]);

      // append the svg canvas to the page
      var svg = d3.select("#SankeyInter").append("svg") //will select the id of cahrt from index.html ln:135 --> # looks for the id= from html
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g") //group everything on the vancas together.  will edit down on ln38 below
          .attr("transform", 
                "translate(" + margin.left + "," + margin.top + ") scale(1,-1) translate(" + 0 + "," + -height + ")");

      // Set the sankey diagram properties
      var sankey = d3Sankey() //calling the function
          .nodeWidth(25)
          .nodePadding(0)
          .size([width, height]);

      var path = sankey.link(); //sankey.link() is something happening in sankey.js 

      // load the data
      var graph = {
      "nodes":[
        {"name":"GradB","node":0,"month":"RandSear","color":"#a6cee3",valueN:this.RandomSearLoc},
        {"name":"RF","node":1,"month":"RandSear","color":"#b15928",valueN:this.RandomSearLoc},
        {"name":"MLP","node":2,"month":"RandSear","color":"#fb9a99",valueN:this.RandomSearLoc},
        {"name":"LR","node":3,"month":"RandSear","color":"#fdbf6f",valueN:this.RandomSearLoc},
        {"name":"KNN","node":4,"month":"RandSear","color":"#ff7f00",valueN:this.RandomSearLoc},
        {"name":"Random Search \u2014 S0","node":5,"month":"RandSear","color":"#ffffff",valueN:this.RandomSearLoc},
        {"name":"GradB","node":6,"month":"Crossover_S1","color":"#a6cee3",valueN:this.RandomSearLoc/2},
        {"name":"RF","node":7,"month":"Crossover_S1","color":"#b15928",valueN:this.RandomSearLoc/2},
        {"name":"MLP","node":8,"month":"Crossover_S1","color":"#fb9a99",valueN:this.RandomSearLoc/2},
        {"name":"LR","node":9,"month":"Crossover_S1","color":"#fdbf6f",valueN:this.RandomSearLoc/2},
        {"name":"KNN","node":10,"month":"Crossover_S1","color":"#ff7f00",valueN:this.RandomSearLoc/2},
        {"name":"(M) Mutate \u2014 S1","node":11,"month":"Crossover_S1","color":"#ffffff",valueN:this.RandomSearLoc/2},
        {"name":"GradB","node":12,"month":"Mutate_S1","color":"#a6cee3",valueN:this.RandomSearLoc/2},
        {"name":"RF","node":13,"month":"Mutate_S1","color":"#b15928",valueN:this.RandomSearLoc/2},
        {"name":"MLP","node":14,"month":"Mutate_S1","color":"#fb9a99",valueN:this.RandomSearLoc/2},
        {"name":"LR","node":15,"month":"Mutate_S1","color":"#fdbf6f",valueN:this.RandomSearLoc/2},
        {"name":"KNN","node":16,"month":"Mutate_S1","color":"#ff7f00",valueN:this.RandomSearLoc/2},
        {"name":"(C) Crossover \u2014 S1","node":17,"month":"Mutate_S1","color":"#ffffff",valueN:this.RandomSearLoc/2},
        {"name":"GradB","node":18,"month":"Crossover_S2","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":19,"month":"Crossover_S2","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":20,"month":"Crossover_S2","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":21,"month":"Crossover_S2","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":22,"month":"Crossover_S2","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Mutate \u2014 S2 (M)","node":23,"month":"Crossover_S2","color":"#ffffff",valueN:this.RandomSearLoc/4},
        {"name":"GradB","node":24,"month":"Mutate_S2","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":25,"month":"Mutate_S2","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":26,"month":"Mutate_S2","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":27,"month":"Mutate_S2","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":28,"month":"Mutate_S2","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Crossover \u2014 S2 (M)","node":29,"month":"Mutate_S2","color":"#ffffff",valueN:this.RandomSearLoc/4},
        {"name":"GradB","node":30,"month":"Crossover_S2_Prime","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":31,"month":"Crossover_S2_Prime","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":32,"month":"Crossover_S2_Prime","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":33,"month":"Crossover_S2_Prime","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":34,"month":"Crossover_S2_Prime","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Mutate \u2014 S2 (C)","node":35,"month":"Crossover_S2_Prime","color":"#ffffff",valueN:this.RandomSearLoc/4},
        {"name":"GradB","node":36,"month":"Mutate_S2_Prime","color":"#a6cee3",valueN:this.RandomSearLoc/4},
        {"name":"RF","node":37,"month":"Mutate_S2_Prime","color":"#b15928",valueN:this.RandomSearLoc/4},
        {"name":"MLP","node":38,"month":"Mutate_S2_Prime","color":"#fb9a99",valueN:this.RandomSearLoc/4},
        {"name":"LR","node":39,"month":"Mutate_S2_Prime","color":"#fdbf6f",valueN:this.RandomSearLoc/4},
        {"name":"KNN","node":40,"month":"Mutate_S2_Prime","color":"#ff7f00",valueN:this.RandomSearLoc/4},
        {"name":"Crossover \u2014 S2 (C)","node":41,"month":"Mutate_S2_Prime","color":"#ffffff",valueN:this.RandomSearLoc/4},
        ],

        "links":[
        {"source":5,"target":11,"value":0},
        {"source":5,"target":17,"value":0},
        {"source":0,"target":6,"value":valuesLoc[6]},
        {"source":0,"target":12,"value":valuesLoc[12]},
        {"source":1,"target":7,"value":valuesLoc[7]},
        {"source":1,"target":13,"value":valuesLoc[13]},
        {"source":2,"target":8,"value":valuesLoc[8]},
        {"source":2,"target":14,"value":valuesLoc[14]},
        {"source":3,"target":9,"value":valuesLoc[9]},
        {"source":3,"target":15,"value":valuesLoc[15]},
        {"source":4,"target":10,"value":valuesLoc[10]},
        {"source":4,"target":16,"value":valuesLoc[16]},
        {"source":11,"target":23,"value":0},
        {"source":11,"target":35,"value":0},
        {"source":6,"target":18,"value":valuesLoc[18]},
        {"source":6,"target":24,"value":valuesLoc[24]},
        {"source":7,"target":19,"value":valuesLoc[19]},
        {"source":7,"target":25,"value":valuesLoc[25]},
        {"source":8,"target":20,"value":valuesLoc[20]},
        {"source":8,"target":26,"value":valuesLoc[26]},
        {"source":9,"target":21,"value":valuesLoc[21]},
        {"source":9,"target":27,"value":valuesLoc[27]},
        {"source":10,"target":22,"value":valuesLoc[22]},
        {"source":10,"target":28,"value":valuesLoc[28]},
        {"source":17,"target":29,"value":0},
        {"source":17,"target":41,"value":0},
        {"source":12,"target":30,"value":valuesLoc[30]},
        {"source":12,"target":36,"value":valuesLoc[36]},
        {"source":13,"target":31,"value":valuesLoc[31]},
        {"source":13,"target":37,"value":valuesLoc[37]},
        {"source":14,"target":32,"value":valuesLoc[32]},
        {"source":14,"target":38,"value":valuesLoc[38]},
        {"source":15,"target":33,"value":valuesLoc[33]},
        {"source":15,"target":39,"value":valuesLoc[39]},
        {"source":16,"target":34,"value":valuesLoc[34]},
        {"source":16,"target":40,"value":valuesLoc[40]},
      ]}

        sankey.nodes(graph.nodes)
          .links(graph.links)
          .layout(0);
      var colorDiff
      colorDiff = d3v5.scaleSequential(d3v5.interpolatePRGn).domain([-100, 100])
      var percentage = this.percentageOverall
      this.storePreviousPercentage = percentage
      // add in the links
        var link = svg.append("g").selectAll(".link")
            .data(graph.links)
          .enter().append("path")
            .attr("class", "link")
            .attr("d", path) //d??? look it up later 
            .style("stroke",function(d){
              if(d.source.node == 5){
                return "transparent";
              }
              if(d.source.node == 11){
                return "transparent";
              }
              if(d.source.node == 17){
                return "transparent";
              }
              if(d.target.node == 16){
                return colorDiff(percentage[0]);
              } else if(d.target.node == 15){
                return colorDiff(percentage[1]);
              } else if(d.target.node == 14){
                return colorDiff(percentage[2]);
              } else if(d.target.node == 13){
                return colorDiff(percentage[3]);
              } else if(d.target.node == 12){
                return colorDiff(percentage[4]);
              } else if(d.target.node == 10){
                return colorDiff(percentage[5]);
              } else if(d.target.node == 9){
                return colorDiff(percentage[6]);
              } else if(d.target.node == 8){
                return colorDiff(percentage[7]);
              } else if(d.target.node == 7){
                return colorDiff(percentage[8]);
              } else if(d.target.node == 6){
                return colorDiff(percentage[9]);
              } else {
                return "#808080"
              }
            }) 
            .style("stroke-width", function(d) { return Math.max(.5, d.dy); })   //setting the stroke length by the data . d.dy is defined in sankey.js
            .sort(function(a, b) { return b.dy - a.dy; })
            .on("mouseover",linkmouseover)
            .on("mouseout",linkmouseout);  

      // add the link titles
        link.append("svg:title") //this is the mouseover stuff title is an svg element you can use "svg:title" or just "title"
              .text(function(d) {
              return format(d.value); });

      // add in the nodes (creating the groups of the rectanlges)
        var node = svg.append("g").selectAll(".node") 
            .data(graph.nodes)
          .enter().append("g")
            .attr("class", "node")
            .attr("transform", function(d) { 
                return "translate(" + d.x + "," + d.y + ")";
            });
          //.call(d3.behavior.drag()   <---------- THIS IS THE DRAG THING TO REMOVE!!
            //.origin(function(d) { return d; })
            // .on("dragstart", function() {  //<-------- THIS IS MOUSEOVER DRAG CAPABILITIES .on(mousemove) called pointer events, look it up!
            // this.parentNode.appendChild(this); }) 
            // .on("drag", dragmove);
        
      // add the rectangles for the nodes
        node.append("rect")
            .attr("height", function(d) { return d.dy; })
            .attr("width", sankey.nodeWidth(  ))
            .style("fill", function(d) { return d.color; }) //matches name with the colors here! inside the replace is some sort of regex
            // .style("stroke",function(d) { return d3.rgb(d.color).darker(1); }) //line around the box formatting
            // .style("stroke-width",.5)
            .on("mouseover", nodemouseover)
            .on("mouseout", nodemouseout);

        if (this.loop == 0) {
          node.append("foreignObject")
            .attr("x", 28)
            .attr("y", -16)
            .attr("height", 18)
            .attr("width", 40)
            .attr("transform", "scale(1,-1)")
            .append("xhtml:body")
              .html(function(d) {
                if (d.node > 17 && d.node != 23 && d.node != 29 && d.node != 35 && d.node != 41) {
                  return '<input id='+d.node+' type="number" value='+valuesLoc[d.node]+' min="0" max='+Math.floor(startingAxis/(localStep*2))+' required/>'
                }
              });

            $("input[type='number']").change( function(d) {
              valuesLoc[d.target.id] = parseInt(d.target.value)
              EventBus.$emit('changeValues2Run', valuesLoc)
            });
        }
            // .append("title")
            // .text(function(d) { 
            // return d.name + "\n" + format(d.value); });

        // add in the title for the nodes
        node.append("text")
            .attr("x", -6)
            .attr("y", function(d) {
              if (d.node <= 5) {
                return d.dy - 81; 
              } else if (d.node <= 17) {
                return d.dy - 41; 
              } else {
                return d.dy - 21; 
              }
            })
            .attr("dy", ".35em")
            .attr("text-anchor", "end")
            .attr("transform", "scale(1,-1)")
            .text(function(d) { return d.name.replace(/-.*/, ""); })
            .style("font-weight", function(d) {
              if (d.node == 5 || d.node == 11 || d.node == 17 || d.node == 23 || d.node == 29 || d.node == 35 || d.node == 41) {
                return "bold";
              }
            })
            .style("font-size", function(d) {
              if (d.node > 17) {
                return "14.5px";
              }
            })
          .filter(function(d) { return d.x < width / 2; })//positioning left or right of node
            .attr("x", 6 + sankey.nodeWidth())
            .attr("text-anchor", "start");
        // // the function for moving the nodes
        //   function dragmove(d) {
        //     d3.select(this).attr("transform", 
        //         "translate(" + d.x + "," + (
        //                 d.y = Math.max(0, Math.min(height/2 - d.dy, d3.event.y))
        //             ) + ")");
        //     sankey.relayout();
        //     link.attr("d", path);
        //   }


      var status=null;
      function nodemouseover(d){

        d3.selectAll(".link")
            .attr("id", function(i){
              if (i.source.node == d.node || i.target.node == d.node){
                status="clicked";
              } else {
                status=null;
              }
              return status;
          });
          }

      function nodemouseout(d){
        d3.selectAll(".link")
            .attr("id", "unclicked");
          }
      function linkmouseover(d){
        d3.select(this)
            .attr("stroke-opacity",.8);
          }
      function linkmouseout(d){
        d3.select(this)
            .attr("stroke-opacity",.3);
          }

      //select all of our links and set a new stroke color on the conditioan that the value is =.01. 
      d3.selectAll(".link")
            .style("stroke-opacity", function(d){ 
                    if(d.value == 0.01) return 0;
                    });

      svg.append("g")
        .append("text")
        .text("Number of Models")
        .attr("font-family","sans-serif")
        .attr("font-size",18.5)
        .attr("fill","black")
        .attr("x",20)
        .attr("y",30)
        .attr("transform", 
                "translate(" + -45 + "," + 0 + ") scale(1,-1) translate(" + 0 + "," + -(height) + ") rotate(-90 150 150)");

      //y axis
        // svg.append("g")
        //     .call(yAxis)
        //     .attr("class", "axis")
        //     .attr("transform", 
        //       "translate(" + -45 + "," + 0 + ") scale(1,-1) translate(" + 0 + "," + -(height) + ")");

    },
    SankeyView () {
      var valuesLoc = this.values
      var valuesLocSt2 = this.valuesStage2
      var localStep = 2
      var numberofModels = 6
      var units = "Models";
      var initialModels = this.RandomSearLoc * 5
      //this is the svg canvas attributes: (not buidlign abything just seeting up varaibels)
      var margin = {top: 0, right: 40, bottom: 0, left: 40}, //comma is the equivalent of var : 
          width = 1230 - margin.left - margin.right,
          height = 350 - margin.top - margin.bottom;

      var svg = d3.select("#SankeyInter");
      svg.selectAll("*").remove();

      var formatNumber = d3Format(",.0f"),    // zero decimal places
        format = function(d) { return formatNumber(d) + " " + units; }
      var color = d3.scale.category20b()

      var startingAxis = this.RandomSearLoc
      var axisScale = d3.scale.linear()
                        .domain([startingAxis*6,0])
                        .range([0, height]);

      //Create the Axis
      var yAxis = d3.svg.axis()
                    .scale(axisScale)
                    .orient("left")
                    .ticks(10);



      // var lossScale = d3.scale.linear()
      //                   .domain([.95,1,1.05])
      //                   .range(["red","black","green"]);

      // append the svg canvas to the page
      var svg = d3.select("#SankeyInter").append("svg") //will select the id of cahrt from index.html ln:135 --> # looks for the id= from html
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g") //group everything on the vancas together.  will edit down on ln38 below
          .attr("transform", 
                "translate(" + margin.left + "," + margin.top + ") scale(1,-1) translate(" + 0 + "," + -height + ")");

      // Set the sankey diagram properties
      var sankey = d3Sankey() //calling the function
          .nodeWidth(25)
          .nodePadding(0)
          .size([width, height]);

      var path = sankey.link(); //sankey.link() is something happening in sankey.js 

      var graph = {
        "nodes":[
        {"name":"GradB","node":0,"month":"RandSear","color":"#a6cee3", valueN: this.RandomSearLoc},
        {"name":"RF","node":1,"month":"RandSear","color":"#b15928", valueN: this.RandomSearLoc},
        {"name":"MLP","node":2,"month":"RandSear","color":"#fb9a99", valueN: this.RandomSearLoc},
        {"name":"LR","node":3,"month":"RandSear","color":"#fdbf6f", valueN: this.RandomSearLoc},
        {"name":"KNN","node":4,"month":"RandSear","color":"#ff7f00", valueN: this.RandomSearLoc},
        {"name":"Random Search \u2014 S0","node":5,"month":"RandSear","color":"#ffffff", valueN: this.RandomSearLoc},
        {"name":"GradB","node":6,"month":"Crossover_S1","color":"#a6cee3", valueN: this.RandomSearLoc/2},
        {"name":"RF","node":7,"month":"Crossover_S1","color":"#b15928", valueN: this.RandomSearLoc/2},
        {"name":"MLP","node":8,"month":"Crossover_S1","color":"#fb9a99", valueN: this.RandomSearLoc/2},
        {"name":"LR","node":9,"month":"Crossover_S1","color":"#fdbf6f", valueN: this.RandomSearLoc/2},
        {"name":"KNN","node":10,"month":"Crossover_S1","color":"#ff7f00", valueN: this.RandomSearLoc/2},
        {"name":"(M) Mutate \u2014 S1","node":11,"month":"Crossover_S1","color":"#ffffff", valueN: this.RandomSearLoc/2},
        {"name":"GradB","node":12,"month":"Mutate_S1","color":"#a6cee3", valueN: this.RandomSearLoc/2},
        {"name":"RF","node":13,"month":"Mutate_S1","color":"#b15928", valueN: this.RandomSearLoc/2},
        {"name":"MLP","node":14,"month":"Mutate_S1","color":"#fb9a99", valueN: this.RandomSearLoc/2},
        {"name":"LR","node":15,"month":"Mutate_S1","color":"#fdbf6f", valueN: this.RandomSearLoc/2},
        {"name":"KNN","node":16,"month":"Mutate_S1","color":"#ff7f00", valueN: this.RandomSearLoc/2},
        {"name":"(C) Crossover \u2014 S1","node":17,"month":"Mutate_S1","color":"#ffffff", valueN: this.RandomSearLoc/2},
        ],

        "links":[
        {"source":5,"target":11,"value":0},
        {"source":5,"target":17,"value":0},
        {"source":0,"target":6,"value":valuesLoc[6]},
        {"source":0,"target":12,"value":valuesLoc[12]},
        {"source":1,"target":7,"value":valuesLoc[7]},
        {"source":1,"target":13,"value":valuesLoc[13]},
        {"source":2,"target":8,"value":valuesLoc[8]},
        {"source":2,"target":14,"value":valuesLoc[14]},
        {"source":3,"target":9,"value":valuesLoc[9]},
        {"source":3,"target":15,"value":valuesLoc[15]},
        {"source":4,"target":10,"value":valuesLoc[10]},
        {"source":4,"target":16,"value":valuesLoc[16]},
      ]}

        sankey.nodes(graph.nodes)
          .links(graph.links)
          .layout(0);

      // add in the links
        var link = svg.append("g").selectAll(".link")
            .data(graph.links)
          .enter().append("path")
            .attr("class", "link")
            .attr("d", path) //d??? look it up later 
            .style("stroke",function(d){
            if(d.source.node == 5){
              return "transparent";
            } else {
              return "#808080"
            }
            })
            .style("stroke-width", function(d) { return Math.max(.5, d.dy); })   //setting the stroke length by the data . d.dy is defined in sankey.js
            .sort(function(a, b) { return b.dy - a.dy; })
            .on("mouseover",linkmouseover)
            .on("mouseout",linkmouseout);  

      // add the link titles
        link.append("svg:title") //this is the mouseover stuff title is an svg element you can use "svg:title" or just "title"
              .text(function(d) {
              return format(d.value); });

      // add in the nodes (creating the groups of the rectanlges)
        var node = svg.append("g").selectAll(".node") 
            .data(graph.nodes)
          .enter().append("g")
            .attr("class", "node")
            .attr("transform", function(d) { 
                return "translate(" + d.x + "," + d.y + ")";
            });
          //.call(d3.behavior.drag()   <---------- THIS IS THE DRAG THING TO REMOVE!!
            //.origin(function(d) { return d; })
            // .on("dragstart", function() {  //<-------- THIS IS MOUSEOVER DRAG CAPABILITIES .on(mousemove) called pointer events, look it up!
            // this.parentNode.appendChild(this); }) 
            // .on("drag", dragmove);
        
      // add the rectangles for the nodes
        node.append("rect")
            .attr("height", function(d) { return d.dy; })
            .attr("width", sankey.nodeWidth(  ))
            .style("fill", function(d) { return d.color; }) //matches name with the colors here! inside the replace is some sort of regex
            // .style("stroke",function(d) { return d3.rgb(d.color).darker(1); }) //line around the box formatting
            // .style("stroke-width",.5)
            .on("mouseover", nodemouseover)
            .on("mouseout", nodemouseout);

        if (this.loop == 0) {
          node.append("foreignObject")
            .attr("x", 28)
            .attr("y", -24)
            .attr("height", 18)
            .attr("width", 40)
            .attr("transform", "scale(1,-1)")
            .append("xhtml:body")
              .html(function(d) {
                if (d.node > 5 && d.node != 11 && d.node != 17) {
                  return '<input id='+d.node+' type="number" value='+valuesLoc[d.node]+' min="0" max='+Math.floor(startingAxis/localStep)+' required/>'
                }
              });

            $("input[type='number']").change( function(d) {
              valuesLoc[d.target.id] = parseInt(d.target.value)
              valuesLocSt2[d.target.id] = parseInt(d.target.value)
              EventBus.$emit('changeValues', valuesLoc)
              EventBus.$emit('changeValues2', valuesLocSt2)
            });
        }
            // .append("title")
            // .text(function(d) { 
            // return d.name + "\n" + format(d.value); });

        // add in the title for the nodes
        node.append("text")
            .attr("x", -6)
            .attr("y", function(d) {
              if (d.node <= 5) {
                return d.dy - 81; 
              } else {
                return d.dy - 41; 
              }
            })
            .attr("dy", ".35em")
            .attr("text-anchor", "end")
            .attr("transform", "scale(1,-1)")
            .text(function(d) { return d.name.replace(/-.*/, ""); })
            .style("font-weight", function(d) {
              if (d.node == 5 || d.node == 11 || d.node == 17) {
                return "bold";
              }
            })
          .filter(function(d) { return d.x < width / 2; })//positioning left or right of node
            .attr("x", 6 + sankey.nodeWidth())
            .attr("text-anchor", "start");


      var status=null;
      function nodemouseover(d){

        d3.selectAll(".link")
            .attr("id", function(i){
              if (i.source.node == d.node || i.target.node == d.node){
                status="clicked";
              } else {
                status=null;
              }
              return status;
          });
          }

      function nodemouseout(d){
        d3.selectAll(".link")
            .attr("id", "unclicked");
          }
      function linkmouseover(d){
        d3.select(this)
            .attr("stroke-opacity",.8);
          }
      function linkmouseout(d){
        d3.select(this)
            .attr("stroke-opacity",.3);
          }

      //select all of our links and set a new stroke color on the conditioan that the value is =.01. 
      d3.selectAll(".link")
            .style("stroke-opacity", function(d){ 
                    if(d.value == 0.01) return 0;
                    });

      svg.append("g")
        .append("text")
        .text("Number of Models")
        .attr("font-family","sans-serif")
        .attr("font-size",18.5)
        .attr("fill","black")
        .attr("x",20)
        .attr("y",30)
        .attr("transform", 
                "translate(" + -45 + "," + 0 + ") scale(1,-1) translate(" + 0 + "," + -(height) + ") rotate(-90 150 150)");

    },
    
  },
  mounted() {
    //EventBus.$on('emittedEventCallingSankeyLegend', this.LegendStable)

    EventBus.$on('emittedEventCallingSankeyStage2', this.SankeyViewStage2)
    EventBus.$on('emittedEventCallingSankeyStage3', this.SankeyViewStage3)

    EventBus.$on('changeValues', data => { this.values = data; })
    EventBus.$on('changeValues', this.SankeyView )

    EventBus.$on('changeValues2', data => { this.valuesStage2 = data})
    EventBus.$on('changeValues2Run', this.SankeyViewStage2)

    EventBus.$on('SendtheChangeinRangePos', data => { this.RandomSearLoc = data })

    EventBus.$on('emittedEventCallingSankey', this.SankeyView)

    EventBus.$on('Responsive', data => {
    this.WH = data})
    EventBus.$on('ResponsiveandChange', data => {
    this.WH = data})

    EventBus.$on('SendPerformance', data => {
    this.PerF = data})
    EventBus.$on('SendPerformanceCM', data => {
    this.PerFCM = data})
    EventBus.$on('SendSank', this.computePerformanceDiff)
    EventBus.$on('SendSankS', this.computePerformanceDiffS)
    EventBus.$on('SendStoredEnsembleHist', data => { this.storedEnsem = data })
    EventBus.$on('SendStoredCMHist', data => { this.storedCM = data })

    // reset the views
    EventBus.$on('resetViews', this.reset)
  }
}
</script>

<style>

.input {
  width: 35px !important;
  height: 18px !important;
}

.node text {
	pointer-events: none;
	text-shadow: 0 1px 0 #fff;
	}

.link {
	fill: none;
	stroke-opacity: .3;
	}

.link:hover {
	stroke-opacity: .8;
	}

#clicked {
  stroke-opacity: .8;
}

.axis path,
.axis line {
  fill: none;
  stroke: #808080;
  shape-rendering: crispEdges;
  margin-left:60px;
}
.axis text {
  font-family: sans-serif;
  font-size: 11px;
}       
</style>