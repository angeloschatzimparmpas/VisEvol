<template>
  <div style="margin: -20px 0 -20px 0 !important">
    <div class="row align-items-center" style="margin: 0 0 -25px 0 !important">
      <div class="col-lg-4"><p>Random search:</p></div>
      <div class="col-lg-6"><div id="slider-stepPos"></div></div>
      <div class="col-lg-2"><p id="value-stepPos"></p></div>
    </div>
    <div class="row align-items-center" style="margin: 0 0 -25px 0 !important">
      <div class="col-lg-4"><p>Cross validation:</p></div>
      <div class="col-lg-6"><div id="slider-stepNeg"></div></div>
      <div class="col-lg-2"><p id="value-stepNeg"></p></div>
    </div>
  </div>
</template>

<script>
import { EventBus } from '../main.js'
import { sliderBottom } from 'd3-simple-slider'
import * as d3Base from 'd3'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base, { sliderBottom })

export default {
  name: 'GlobalParamController',
  data () {
    return {
    }
  },
  methods: {
    InitSliders () { 
      var svg = d3.select("#slider-stepPos");
      svg.selectAll("*").remove();

      var svg = d3.select("#slider-stepNeg");
      svg.selectAll("*").remove();

      var dataCorrect = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0];
      var dataWrong = [5.0, 10.0, 15.0];

        var sliderStepPos = d3
          .sliderBottom()
          .min(d3.min(dataCorrect))
          .max(d3.max(dataCorrect))
          .width(200)
          .tickFormat(d3.format(".0f"))
          .ticks(6)
          .step(50)
          .default(100.0)
          .on('onchange', val => {
            d3.select('p#value-stepPos').text(d3.format(".0f")(val));
            EventBus.$emit('SendtheChangeinRangePos', d3.format(".0f")(val))
          });

        var gStepPos = d3
          .select('div#slider-stepPos')
          .append('svg')
          .attr('width', 500)
          .attr('height', 100)
          .append('g')
          .attr('transform', 'translate(30,30)');

        gStepPos.call(sliderStepPos);

        d3.select('p#value-stepPos').text(d3.format(".0f")(sliderStepPos.value()));

        var sliderStepNeg = d3
          .sliderBottom()
          .min(d3.min(dataWrong))
          .max(d3.max(dataWrong))
          .width(200)
          .tickFormat(d3.format(".0f"))
          .ticks(3)
          .step(5)
          .default(5.0)
          .on('onchange', val => {
            d3.select('p#value-stepNeg').text(d3.format(".0f")(val));
            EventBus.$emit('SendtheChangeinRangeNeg', d3.format(".0f")(val))
          });

        var gStepNeg = d3
          .select('div#slider-stepNeg')
          .append('svg')
          .attr('width', 500)
          .attr('height', 100)
          .append('g')
          .attr('transform', 'translate(30,30)');

        gStepNeg.call(sliderStepNeg);

        d3.select('p#value-stepNeg').text(d3.format(".0f")(sliderStepNeg.value()));
    },
  },
  mounted () {
    this.InitSliders()
    EventBus.$on('reset', this.InitSliders)
  },
}
</script>
