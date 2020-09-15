<template>
  <div>
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
  name: "AlgorithmsController",
  data () {
    return {

    }
  },
  methods: {
    reset () {

    },
}