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
      step: 2,
      values: [0,0,0,0,0,0,50,50,50,50,50,0,50,50,50,50,50,0],
      loop: 0
    }
  },
  methods: {
    reset () {
      var svg = d3.select("#SankeyInter");
      svg.selectAll("*").remove();
    },
    SankeyView () {
    var valuesLoc = this.values
    var localStep = this.step
    var numberofModels = 6
    var units = "Models";
    var initialModels = this.RandomSearLoc * 5
    var months = [{month:"RandSear",value:initialModels,loss:null},
                  {month:"Crossover",value:250,loss:null},
                  {month:"Mutate",value:250,loss:null}];
    //this is the svg canvas attributes: (not buidlign abything just seeting up varaibels)
    var margin = {top: 10, right: 40, bottom: 10, left: 100}, //comma is the equivalent of var : 
        width = 1200 - margin.left - margin.right,
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
    // Fix that!
    d3.select("svg")
      .append("text")
      .text("Player Count")
      .attr("x",30)
      .attr("y",17)
      .attr("font-family","Pontano Sans")
      .attr("font-size",18.5)
      .attr("fill","black")
      .attr("transform", function(d){ 
          return "translate(" + 0 + "," + 0 + ") rotate(-90 150 150)";});

    // Set the sankey diagram properties
    var sankey = d3Sankey() //calling the function
        .nodeWidth(25)
        .nodePadding(0)
        .size([width, height]);

    var path = sankey.link(); //sankey.link() is something happening in sankey.js 

    // svg.selectAll("text.values")
    //   .data(months)
    //   .enter()
    //   .append("text")
    //   .text(function(d){return formatNumber(d.value)})
    //   .attr("class", "innerText")
    //   .attr("x",function(d,i){return i*89-margin.left-5})
    //   .attr("y",20)
    //   .attr("transform", function(d){ 
    //           return "translate(" + margin.left + "," + margin.top + ") scale(1,-1) translate(" + 0 + "," + -(d.value/10+15) + ")";});

    // svg.selectAll("text.loss")
    //   .data(months)
    //   .enter()
    //   .append("text")
    //   .text(function(d){return d.loss})
    //   .attr("class", "innerText")
    //   .attr("x",function(d,i){return i*89-margin.left-5})
    //   .attr("y",20)
    //   .attr("fill",function(d){ return lossScale(d.loss)})
    //   .attr("transform", function(d){ 
    //           return "translate(" + margin.left + "," + margin.top + ") scale(1,-1) translate(" + 0 + "," + -(d.value/10-5) + ")";});

    // svg.selectAll("text.months")
    //   .data(months)
    //   .enter()
    //   .append("text")
    //   .attr("class", "innerText")
    //   .text(function(d){return d.month})
    //   .attr("x",function(d,i){return i*89-margin.left-10})
    //   .attr("y",20)
    //   .attr("transform", 
    //           "translate(" + margin.left + "," + margin.top + ") scale(1,-1) translate(" + 0 + "," + margin.bottom + ")");

    // load the data
    var graph = {
      "nodes":[
      {"name":"GradB","node":0,"month":"RandSear","color":"#e41a1c","dh":height/numberofModels},
      {"name":"RF","node":1,"month":"RandSear","color":"#377eb8","dh":height/numberofModels},
      {"name":"MLP","node":2,"month":"RandSear","color":"#4daf4a","dh":height/numberofModels},
      {"name":"LR","node":3,"month":"RandSear","color":"#984ea3","dh":height/numberofModels},
      {"name":"KNN","node":4,"month":"RandSear","color":"#ff7f00","dh":height/numberofModels},
      {"name":"Random search","node":5,"month":"RandSear","color":"#ffffff","dh":height/numberofModels},
      {"name":"GradB","node":6,"month":"Crossover","color":"#e41a1c","dh":height/(numberofModels*localStep)},
      {"name":"RF","node":7,"month":"Crossover","color":"#377eb8","dh":height/(numberofModels*localStep)},
      {"name":"MLP","node":8,"month":"Crossover","color":"#4daf4a","dh":height/(numberofModels*localStep)},
      {"name":"LR","node":9,"month":"Crossover","color":"#984ea3","dh":height/(numberofModels*localStep)},
      {"name":"KNN","node":10,"month":"Crossover","color":"#ff7f00","dh":height/(numberofModels*localStep)},
      {"name":"Mutate","node":11,"month":"Crossover","color":"#ffffff","dh":height/(numberofModels*localStep)},
      {"name":"GradB","node":12,"month":"Mutate","color":"#e41a1c","dh":height/(numberofModels*localStep)},
      {"name":"RF","node":13,"month":"Mutate","color":"#377eb8","dh":height/(numberofModels*localStep)},
      {"name":"MLP","node":14,"month":"Mutate","color":"#4daf4a","dh":height/(numberofModels*localStep)},
      {"name":"LR","node":15,"month":"Mutate","color":"#984ea3","dh":height/(numberofModels*localStep)},
      {"name":"KNN","node":16,"month":"Mutate","color":"#ff7f00","dh":height/(numberofModels*localStep)},
      {"name":"Crossover","node":17,"month":"Crossover","color":"#ffffff","dh":height/(numberofModels*localStep)},
      ],

      "links":[
      {"source":5,"target":11,"value":50,"dh":height/(numberofModels*localStep)*(250/(valuesLoc[6]+valuesLoc[7]+valuesLoc[8]+valuesLoc[9]+valuesLoc[10]))},
      {"source":5,"target":17,"value":50,"dh":height/(numberofModels*localStep)*(250/(valuesLoc[12]+valuesLoc[13]+valuesLoc[14]+valuesLoc[15]+valuesLoc[16]))},
      {"source":0,"target":6,"value":valuesLoc[6],"dh":height/(numberofModels*localStep)*(valuesLoc[6]/50)},
      {"source":0,"target":12,"value":valuesLoc[12],"dh":height/(numberofModels*localStep)*(valuesLoc[12]/50)},
      {"source":1,"target":7,"value":valuesLoc[7],"dh":height/(numberofModels*localStep)*(valuesLoc[7]/50)},
      {"source":1,"target":13,"value":valuesLoc[13],"dh":height/(numberofModels*localStep)*(valuesLoc[13]/50)},
      {"source":2,"target":8,"value":valuesLoc[8],"dh":height/(numberofModels*localStep)*(valuesLoc[8]/50)},
      {"source":2,"target":14,"value":valuesLoc[14],"dh":height/(numberofModels*localStep)*(valuesLoc[14]/50)},
      {"source":3,"target":9,"value":valuesLoc[9],"dh":height/(numberofModels*localStep)*(valuesLoc[9]/50)},
      {"source":3,"target":15,"value":valuesLoc[15],"dh":height/(numberofModels*localStep)*(valuesLoc[15]/50)},
      {"source":4,"target":10,"value":valuesLoc[10],"dh":height/(numberofModels*localStep)*(valuesLoc[10]/50)},
      {"source":4,"target":16,"value":valuesLoc[16],"dh":height/(numberofModels*localStep)*(valuesLoc[16]/50)},
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
          }})
          .style("stroke-width", function(d) { return Math.max(.5, d.dh); })   //setting the stroke length by the data . d.dh is defined in sankey.js
          .sort(function(a, b) { return b.dh - a.dh; })
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
          .attr("height", function(d) { return d.dh; })
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
            console.log(valuesLoc)
            EventBus.$emit('changeValues', valuesLoc)
            // your code
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
              return d.dh - 81; 
            } else {
              return d.dh - 41; 
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
          .attr("stroke-opacity",.5);
        }
    function linkmouseout(d){
      d3.select(this)
          .attr("stroke-opacity",.05);
        }

    //select all of our links and set a new stroke color on the conditioan that the value is =.01. 
    d3.selectAll(".link")
          .style("stroke-opacity", function(d){ 
                  if(d.value == 0.01) return 0;
                  });

    //y axis
      svg.append("g")
          .call(yAxis)
          .attr("class", "axis")
          .attr("transform", 
            "translate(" + -45 + "," + 0 + ") scale(1,-1) translate(" + 0 + "," + -(height) + ")");

  },
  },
  mounted() {
    EventBus.$on('changeValues', data => { this.values = data })
    EventBus.$on('changeValues', this.SankeyView )

    EventBus.$on('SendtheChangeinRangePos', data => { this.RandomSearLoc = data })

    EventBus.$on('emittedEventCallingSankey', this.SankeyView)

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
	stroke: #000;
	stroke-opacity: .05;
	}

.link:hover {
	stroke-opacity: .5;
	}

.node text {
	pointer-events: none;
	text-shadow: 0 1px 0 #fff;
	}

.link {
	fill: none;
	stroke: #000;
	stroke-opacity: .05;
	}

.link:hover {
	stroke-opacity: .5;
	}

#clicked {
                    stroke-opacity: .5 !important;
              }
/*          #unclicked {
                    stroke-opacity: .05;
              }*/
          .axis path,
          .axis line {
              fill: none;
              stroke: black;
              shape-rendering: crispEdges;
              margin-left:60px;
          }
          .axis text {
              font-family: sans-serif;
              font-size: 11px;
          }       
</style>