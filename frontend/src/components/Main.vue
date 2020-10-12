<!-- Main Visualization View -->

<template>
<body>
    <b-container fluid class="bv-example-row">
      <b-row class="md-3">
        <b-col cols="3" >
          <mdb-card>
            <mdb-card-header color="primary-color" tag="h5" class="text-center">Data Sets and Validation Metrics Manager</mdb-card-header>
              <mdb-card-body>
                <mdb-card-text class="text-left" style="font-size: 18.5px;">
                  <PerformanceMetrics/>
                  <GlobalParamController/>
                  <DataSetExecController/>
                </mdb-card-text>
              </mdb-card-body>
          </mdb-card>
        </b-col>
        <b-col cols="6">
          <mdb-card>
            <mdb-card-header color="primary-color" tag="h5" class="text-center"><span class="float-left"><font-awesome-icon icon="arrow-alt-circle-right"/> {{ Status }}</span>Process Tracker and Algorithms/Models Selector<span class="badge badge-primary badge-pill float-right">Active<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">1&2</span></span></mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 357px">
                <History/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
        </b-col>
        <b-col cols="3">
            <mdb-card >
              <mdb-card-header color="primary-color" tag="h5" class="text-center">Overall Performance for Each Algorithm/Model<span class="badge badge-primary badge-pill float-right">Active<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">1&2</span></span></mdb-card-header>
              <mdb-card-body>
                <mdb-card-text class="text-left" style="font-size: 18.5px; min-height: 357px">
                  <AlgorithmsController/>
                </mdb-card-text>
              </mdb-card-body>
            </mdb-card>
        </b-col>
      </b-row>
    <b-row class="md-3">
        <b-col cols="6">
          <mdb-card style="margin-top: 15px;">
            <mdb-card-header color="primary-color" tag="h5" class="text-center">Hyper-Parameters' Space
              [Sel: {{OverSelLength}} / All: {{OverAllLength}}]<small class="float-right"></small><span class="badge badge-info badge-pill float-right">Projection<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">1</span></span>
            </mdb-card-header>
            <mdb-card-body>
              <mdb-card-text class="text-center"  style="min-height: 434px">
                <HyperParameterSpace/>
              </mdb-card-text>
            </mdb-card-body>
          </mdb-card>
        </b-col>
        <b-col cols="6">
          <mdb-card style="margin-top: 15px;">
            <mdb-card-header color="primary-color" tag="h5" class="text-center"><span class="float-left"><Knowledge/></span>Majority-Voting Ensemble
              [Sel: {{OverSelLengthCM}} / All: {{OverAllLengthCM}}]<small class="float-right"></small><span class="badge badge-info badge-pill float-right">Projection<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">2</span></span>
              </mdb-card-header>
              <mdb-card-body>
                <mdb-card-text class="text-center"  style="min-height: 430px">
                  <Ensemble/>
                </mdb-card-text>
              </mdb-card-body>
          </mdb-card>
        </b-col>
      </b-row>
      <b-row class="md-3">
        <b-col cols="3">
          <mdb-card style="margin-top: 15px;">
            <mdb-card-header color="primary-color" tag="h5" class="text-center">Performance for Each Validation Metric<span class="badge badge-primary badge-pill float-right">Active<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">{{projectionID_B}}</span></span>
              </mdb-card-header>
              <mdb-card-body>
                <mdb-card-text class="text-center"  style="min-height: 270px">
                  <ValidationController/>
                </mdb-card-text>
              </mdb-card-body>
          </mdb-card>
        </b-col>
        <b-col cols="6">
          <mdb-card style="margin-top: 15px;">
            <mdb-card-header color="primary-color" tag="h5" class="text-center"><span class="float-left"><font-awesome-icon id="mode0" icon="stop-circle"/><font-awesome-icon id="mode1" style="display: none" icon="play-circle"/> {{ MeanShift }}</span>Predictive Results for Each Data Instance<span class="badge badge-primary badge-pill float-right">Active<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">{{projectionID_A}}</span></span>
              </mdb-card-header>
              <mdb-card-body>
                <mdb-card-text class="text-center"  style="min-height: 270px">
                  <Predictions/>
                </mdb-card-text>
              </mdb-card-body>
          </mdb-card>
        </b-col>
        <b-col cols="3">
          <mdb-card style="margin-top: 15px;">
            <mdb-card-header color="primary-color" tag="h5" class="text-center"><span class="float-left"><font-awesome-icon icon="calculator" /></span>Performance for Majority-Voting Ensemble<span class="badge badge-primary badge-pill float-right">Active<span class="badge badge-light" style="margin-left:4px; margin-bottom:1px">2</span></span>
              </mdb-card-header>
              <mdb-card-body>
                <mdb-card-text class="text-center"  style="min-height: 270px">   
                  <VotingResults/>
                </mdb-card-text>
              </mdb-card-body>
          </mdb-card>
        </b-col>
      </b-row>
    </b-container>
    <div class="w3-container">
    <div id="myModal" class="w3-modal" style="position: fixed;">
      <div class="w3-modal-content w3-card-4 w3-animate-zoom">
        <header class="w3-container w3-blue"> 
        <h3 style="display:inline-block; font-size: 22px; margin-top: 15px; margin-bottom:15px">Majority-Voting Ensemble Extraction (using Cryo)</h3>
        </header>
        <Export/>
        <div class="w3-container w3-light-grey w3-padding">
        <button style="float: right; margin-top: -3px; margin-bottom: -3px"
          id="closeModal" class="w3-button w3-right w3-white w3-border" 
          v-on:click="closeModalFun">
          <font-awesome-icon icon="window-close" />
          {{ valuePickled }}
          </button>
        </div>
        </div>
      </div>
    </div>
  </body>
</template>

<script>

import Vue from 'vue'
import DataSetExecController from './DataSetExecController.vue'
import PerformanceMetrics from './PerformanceMetrics.vue'
import AlgorithmsController from './AlgorithmsController.vue'
import ValidationController from './ValidationController.vue'
import HyperParameterSpace from './HyperParameterSpace.vue'
import GlobalParamController from './GlobalParamController'
import Ensemble from './Ensemble.vue'
import Knowledge from './Knowledge.vue'
import Export from './Export.vue'
import VotingResults from './VotingResults.vue'
import History from './History.vue'
import Predictions from './Predictions.vue'
import axios from 'axios'
import { loadProgressBar } from 'axios-progress-bar'
import 'axios-progress-bar/dist/nprogress.css'
import 'bootstrap-css-only/css/bootstrap.min.css'
import { mdbCard, mdbCardBody, mdbCardText, mdbCardHeader } from 'mdbvue'
import { EventBus } from '../main.js'
import $ from "jquery";
import 'bootstrap';
import * as d3Base from 'd3'
import Papa from 'papaparse'

// attach all d3 plugins to the d3 library
const d3 = Object.assign(d3Base)

export default Vue.extend({
  name: 'Main',
  components: {
    DataSetExecController,
    PerformanceMetrics,
    AlgorithmsController,
    ValidationController,
    HyperParameterSpace,
    GlobalParamController,
    Ensemble,
    Knowledge,
    Export,
    Predictions,
    VotingResults,
    History,
    mdbCard,
    mdbCardBody,
    mdbCardHeader,
    mdbCardText
  },
  data () {
    return {
      Status: " (S) Stage 0",
      MeanShift: "K-means Clustering",
      mode: 0,
      valuePickled: 'Close',
      sankeyCallS: 1,
      CMNumberofModelsOFFICIAL: [0,0,0,0,0,0,50,50,50,50,50,0,50,50,50,50,50,0],
      CMNumberofModels: [0,0,0,0,0,0,5,5,5,5,5,0,5,5,5,5,5,0], // Remove that!
      CMNumberofModelsOFFICIALS2: [0,0,0,0,0,0,50,50,50,50,50,0,50,50,50,50,50,0,25,25,25,25,25,0,25,25,25,25,25,0,25,25,25,25,25,0,25,25,25,25,25,0],
      CMNumberofModelsS2: [0,0,0,0,0,0,5,5,5,5,5,0,5,5,5,5,5,0,2,2,2,2,2,0,2,2,2,2,2,0,2,2,2,2,2,0,2,2,2,2,2,0], // Remove that!
      CurrentStage: 1,
      projectionID_A: 1,
      projectionID_B: 1,
      storeEnsemble: [],
      storeEnsemblePermanently: [],
      PredictSelEnsem: [],
      firstTimeExec: true,
      unselectedRemainingPoints: [],
      unselectedRemainingPointsEnsem: [],
      Collection: 0,
      OverviewResults: 0,
      preDataResults: '',
      DataResults: '',
      keyNow: 1,
      instancesImportance: '',
      RetrieveValueFile: 'biodegC', // this is for the default data set
      ClassifierIDsList: [],
      ClassifierIDsListCM: [],
      SelectedFeaturesPerClassifier: '',
      FinalResults: 0,
      Algorithms: ['KNN','LR','MLP','RF','GradB'],
      selectedAlgorithm: '',
      PerformancePerModel: '',
      PerformanceCheck: '',
      selectedModels_Stack: [],
      selectedAlgorithms: ['KNN','LR','MLP','RF','GradB'],
      parametersofModels: [],
      reset: false,
      brushedBoxPlotUpdate: 0,
      width: 0,
      height: 0,
      combineWH: [],
      basicValuesFact: [1,1,1,1,0,0,0,0],
      sumPerClassifier: [],
      valueSel: 0,
      valueAll: 0,
      OverSelLength: 0,
      OverAllLength: 0,
      OverSelLengthCM: 0,
      OverAllLengthCM: 0,
      modelsUpdate: [],
      AlgorithmsUpdate: [],
      SelectedMetricsForModels: [],
      DataPointsSel: '',
      DataPointsModels: '',
      dataPointsSelfromDataSpace: '',
      userSelectedFilterMain: 'mean',
      actionData: '',
      filterData: '',
      provenanceData: '',
      localFile: '',
      toggleDeepMain: 1,
      keyLoc: 0,
      keyData: true,
      ClassifierIDsListRemaining: [],
      PredictSel: [],
      storeBothEnsCM: [],
      crossVal: '5',
      RandomSear: '100',
    }
  },
  methods: {
    openModalFun () {
      $('#myModal').modal('show')
    },
    closeModalFun () {
      $('#myModal').modal('hide')
    },
    getCollection () {
      this.Collection = this.getCollectionFromBackend()
    },
    getCollectionFromBackend () {
      const path = `http://localhost:5000/data/ClientRequest`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.Collection = response.data.Collection
          EventBus.$emit('emittedEventCallingDataPlot', this.Collection)
          console.log('Collection was overwritten with new data sent by the server!')
        })
        .catch(error => {
          console.log(error)
        })
    },
    getDatafromtheBackEnd () {
      const path = `http://localhost:5000/data/PlotClassifiers`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.OverviewResults = response.data.OverviewResults
          console.log('Server successfully sent all the data related to visualizations!')
          if (this.firstTimeExec) {

            var ModelsLocalInitial = JSON.parse(this.OverviewResults[0])
            EventBus.$emit('SendStoredIDsInitial', ModelsLocalInitial)
            var PerformanceInitial = JSON.parse(this.OverviewResults[1])
            this.mode = JSON.parse(this.OverviewResults[15])
            if (this.mode == 1) {
              this.swapSymbol()
            }
            EventBus.$emit('SendPerformanceInitialAlgs', PerformanceInitial)    

            EventBus.$emit('emittedEventCallingScatterPlot', this.OverviewResults)
            EventBus.$emit('emittedEventCallingSankey')
            //EventBus.$emit('emittedEventCallingSankeyLegend')
            EventBus.$emit('callValidationData', this.OverviewResults)
            EventBus.$emit('callValidation')
            EventBus.$emit('emittedEventCallingGrid', this.OverviewResults)
            EventBus.$emit('emittedEventCallingGridSelection')
            EventBus.$emit('emittedEventCallingInfo', this.OverviewResults)
            EventBus.$emit('LegendPredict')           
            this.storeBothEnsCM[0] = this.OverviewResults
            this.firstTimeExec = false
            EventBus.$emit('callAlgorithhms')
            this.Status = " (S) Stage 1"
          } else {
            var IDsPreviously = JSON.parse(this.OverviewResults[16])   
            var Performance = JSON.parse(this.OverviewResults[1])
            EventBus.$emit('SendModelsAll', IDsPreviously)
            EventBus.$emit('SendPerformance', Performance)
            EventBus.$emit('SendStoredEnsembleHist', this.storeEnsemblePermanently)
            EventBus.$emit('SendStoredEnsemble', this.storeEnsemblePermanently)
            EventBus.$emit('emittedEventCallingCrossoverMutation', this.OverviewResults)
            this.PredictSelEnsem = []
            this.storeBothEnsCM[1] = this.OverviewResults
            this.getFinalResults()
            if (this.sankeyCallS == 1) {
              EventBus.$emit('SendSank')
              EventBus.$emit('emittedEventCallingSankeyStage2')
              this.Status = " (S) Stage 2"
            } else if (this.sankeyCallS == 2){
              EventBus.$emit('SendSankS')
              EventBus.$emit('emittedEventCallingSankeyStage3')
              EventBus.$emit('hideCrossMut')
              this.Status = " (S) Stage \u2014"
            } else {
            }
            EventBus.$emit('emittedEventCallingGrid', this.OverviewResults)
            EventBus.$emit('SendSelectedPointsToServerEvent', this.PredictSelEnsem)
            //EventBus.$emit('emittedEventCallingGridSelection', this.OverviewResults)
            EventBus.$emit('callValidationData', this.OverviewResults)
            EventBus.$emit('callValidation')
            EventBus.$emit('LegendPredictEnsem') 
            EventBus.$emit('callAlgorithhms')
            
          }
        })
        .catch(error => {
          console.log(error)
        })
    },
    getCMComputedData () {
      const path = `http://localhost:5000/data/PlotCrossMutate`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.OverviewResultsCM = response.data.OverviewResultsCM

          var ModelsLocal = JSON.parse(this.OverviewResultsCM[0])
          EventBus.$emit('SendStoredCMHist', ModelsLocal)
          EventBus.$emit('SendStoredIDsInitial', ModelsLocal)
          var PerformanceCM = JSON.parse(this.OverviewResultsCM[1])
          EventBus.$emit('SendPerformanceCM', PerformanceCM)
          EventBus.$emit('SendPerformanceInitialAlgs', PerformanceCM)
          console.log('Server successfully sent all the data related to visualizations for CM!')
          EventBus.$emit('emittedEventCallingScatterPlot', this.OverviewResultsCM)
          this.storeBothEnsCM[0] = this.OverviewResultsCM

          EventBus.$emit('callAlgorithhms')
          EventBus.$emit('SendSelectedPointsUpdateIndicatorCM', [])
          //EventBus.$emit('emittedEventCallingSankey', this.OverviewResultsCM)
          //this.PredictSel = []
          //EventBus.$emit('emittedEventCallingGrid', this.OverviewResultsCM)
          //EventBus.$emit('SendSelectedPointsToServerEvent', this.PredictSel)
          //EventBus.$emit('emittedEventCallingGridSelection', this.OverviewResultsCM)
        })
        .catch(error => {
          console.log(error)
        })
    },
    SendToServerData () {
      const path = `http://127.0.0.1:5000/data/SendtoSeverDataSet`

      const postData = {
        uploadedData: this.localFile
      }
      const axiosConfig = {
      headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
      'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
      }
      }
      axios.post(path, postData, axiosConfig)
      .then(response => {
        console.log('Sent the new uploaded data to the server!')
      })
      .catch(error => {
      console.log(error)
      })
    },
    SelectedPoints () {
      this.OverSelLength = this.ClassifierIDsList.length
      this.SendSelectedIDs()
    },
    SendSelectedIDs () {
      const path = `http://127.0.0.1:5000/data/SendtoSeverSelIDs`
      const postData = {
        predictSelectionIDs: this.ClassifierIDsList
      }
      const axiosConfig = {
      headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
      'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
      }
      }
      axios.post(path, postData, axiosConfig)
      .then(response => {
        console.log('Sent the selected IDs to compute predictions!')
        this.retrievePredictionsSel()
      })
      .catch(error => {
      console.log(error)
      }) 
    },
    retrievePredictionsSel () {
      const path = `http://localhost:5000/data/RetrievePredictions`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.PredictSel = response.data.PredictSel
          console.log('Server successfully sent the predictions!')

          EventBus.$emit('emittedEventCallingGrid', this.storeBothEnsCM[0])
          EventBus.$emit('SendSelectedPointsToServerEvent', this.PredictSel)
        })
        .catch(error => {
          console.log(error)
        })
    },
    SendSelectedIDsEnsemble () {
      const path = `http://127.0.0.1:5000/data/SendtoSeverSelIDsEnsem`
      const postData = {
        predictSelectionIDsCM: this.ClassifierIDsListCM
      }
      const axiosConfig = {
      headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
      'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
      }
      }
      axios.post(path, postData, axiosConfig)
      .then(response => {
        console.log('Sent the selected IDs to compute predictions!')
        this.retrievePredictionsSelEnsemble()
      })
      .catch(error => {
      console.log(error)
      }) 
    },
    retrievePredictionsSelEnsemble () {
      const path = `http://localhost:5000/data/RetrievePredictionsEnsem`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.PredictSelEnsem = response.data.PredictSelEnsem
          console.log('Server successfully sent the predictions!')
          EventBus.$emit('emittedEventCallingGrid', this.storeBothEnsCM[1])
          EventBus.$emit('SendSelectedPointsToServerEvent', this.PredictSelEnsem)
        })
        .catch(error => {
          console.log(error)
        })
    },
    SelectedPointsCM () {
      this.OverSelLengthCM = this.ClassifierIDsListCM.length
      this.SendSelectedIDsEnsemble()
    },
    SendSelectedPointsToServer () {
      if (this.ClassifierIDsListCM === ''){
        this.OverSelLengthCM = 0
        EventBus.$emit('resetViews')
      } else {
        this.OverSelLengthCM = this.ClassifierIDsListCM.length
        const path = `http://127.0.0.1:5000/data/ServerRequestSelPoin`

        const postData = {
          ClassifiersList: this.ClassifierIDsListCM,
        }
        const axiosConfig = {
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
            'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
          }
        }
        axios.post(path, postData, axiosConfig)
          .then(response => {
            console.log('Sent the selected points to the server (scatterplot)!')
            // if (this.keyNow == 0) {
            //   this.OverAllLength = this.ClassifierIDsList.length
            //   EventBus.$emit('GrayOutPoints', this.ClassifierIDsList)
            // } 
            //this.getSelectedModelsMetrics()
            this.getFinalResults()
          })
          .catch(error => {
            console.log(error)
          })
      }
    },
    RemoveFromEnsembleModels () {
      const path = `http://127.0.0.1:5000/data/ServerRemoveFromEnsemble`
      const postData = {
        ClassifiersList: this.ClassifierIDsListRemaining,
      }
      const axiosConfig = {
      headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
      'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
      }
      }
      axios.post(path, postData, axiosConfig)
      .then(response => {
      console.log('Remove from Ensemble (scatterplot)!')
      this.storeEnsemblePermanently = []
      for (let i = 0; i < this.ClassifierIDsListRemaining.length; i++) {
        this.storeEnsemblePermanently.push(this.ClassifierIDsListRemaining[i])
      }
      console.log(this.storeEnsemblePermanently)
      EventBus.$emit('SendSelectedPointsUpdateIndicatorCM', [])
      this.getDatafromtheBackEnd()
      })
      .catch(error => {
      console.log(error)
      })
    },
    updatePredictionsSpace () {
      const path = `http://localhost:5000/data/UpdatePredictionsSpace`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.UpdatePredictions = response.data.UpdatePredictions
          console.log('Updating Predictions Space!')
          if (this.keyNow == 1) {
            EventBus.$emit('InitializeProvenance', this.UpdatePredictions)
            EventBus.$emit('sendKeyScatt', 1)
            EventBus.$emit('GrayOutPoints', this.ClassifierIDsList)
          }
          EventBus.$emit('updatePredictionsSpace', this.UpdatePredictions)
          EventBus.$emit('updateFlagForFinalResults', 0)
          this.getFinalResults()
        })
        .catch(error => {
          console.log(error)
        })
    },
    SendSelectedDataPointsToServer () {
      // set a path from which the server will receive the seleceted predictions points
      const path = `http://127.0.0.1:5000/data/ServerRequestDataPoint`
      // brushing and linking between predictions space and data space 
      EventBus.$emit('updateDataSpaceHighlighting', this.DataPointsSel)

      const postData = {
        DataPointsSel: this.DataPointsSel
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('Sent the selected data points to the server!')
          this.getSelectedDataPointsModels()
        })
        .catch(error => {
          console.log(error)
        })
    },
    getSelectedDataPointsModels () {
      const path = `http://localhost:5000/data/ServerSentDataPointsModel`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.DataPointsModels = response.data.DataPointsModels
          var resultsPerMetricUpdated = JSON.parse(this.DataPointsModels[2])
          console.log('Server successfully sent the new models for the scatterplot!')
          EventBus.$emit('UpdateModelsScatterplot', this.DataPointsModels)
          EventBus.$emit('InitializeMetricsBarChartPrediction', JSON.stringify(resultsPerMetricUpdated))
          EventBus.$emit('UpdateBalanceView', this.DataPointsModels)
        })
        .catch(error => {
          console.log(error)
        })
    },
    getSelectedModelsMetrics () {
      const path = `http://localhost:5000/data/BarChartSelectedModels`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.SelectedMetricsForModels = response.data.SelectedMetricsForModels
          console.log('Server successfully updated barchart for metrics based on selected models!')
          EventBus.$emit('UpdateBarChartperMetric', this.SelectedMetricsForModels)
        })
        .catch(error => {
          console.log(error)
        })
    },
    getFinalResults () {
      this.FinalResults = this.getFinalResultsFromBackend()
    },
    getFinalResultsFromBackend () {
      const path = `http://localhost:5000/data/SendFinalResultsBacktoVisualize`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.FinalResults = response.data.FinalResults
          EventBus.$emit('emittedEventCallingResultsPlot', this.FinalResults)
        })
        .catch(error => {
          console.log(error)
        })
    },
    fileNameSend () {
      const path = `http://127.0.0.1:5000/data/ServerRequest`
      
      const postData = {
        fileName: this.RetrieveValueFile,
        RandomSearch: this.RandomSear,
        CrossValidation: this.crossVal,
        Factors: this.basicValuesFact
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
      .then(response => {
        console.log('File name was sent successfully!')
        this.CMNumberofModelsOFFICIAL = [0,0,0,0,0,0,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,0,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,0]
        this.CMNumberofModelsOFFICIALS2 = [0,0,0,0,0,0,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,0,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,this.RandomSear/2,0,Math.floor(this.RandomSear/4),this.RandomSear/4,Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),0,Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),0,Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),0,Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),Math.floor(this.RandomSear/4),0]
        EventBus.$emit('updateRandomS', this.RandomSear)
        EventBus.$emit('updateStage1', this.CMNumberofModelsOFFICIAL)
        EventBus.$emit('updateStage2', this.CMNumberofModelsOFFICIALS2)
        this.SendAlgorithmsToServer()
      })
      .catch(error => {
        console.log(error)
      })
    },
    SendAlgorithmsToServer () {
      const path = `http://127.0.0.1:5000/data/ServerRequestSelParameters`
      const postData = {
        Algorithms: this.Algorithms,
        Toggle: this.toggleDeepMain
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('Send request to server! Algorithm name was sent successfully!')
          this.factors()
        })
        .catch(error => {
          console.log(error)
        })
    },
    UpdateBarChartFeatures () {
      const path = `http://127.0.0.1:5000/data/FeaturesScoresUpdate`
      const postData = {
        models: this.modelsUpdate,
        algorithms: this.AlgorithmsUpdate
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('Send request to server! Updating Barchart!')
          this.UpdateModelsFeaturePerformance()
        })
        .catch(error => {
          console.log(error)
        })
    },
    UpdateBasedonFeatures () {
      const path = `http://127.0.0.1:5000/data/FeaturesSelection`
        const postData = {
          featureSelection: this.SelectedFeaturesPerClassifier
        }
        const axiosConfig = {
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
            'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
          }
        }
        axios.post(path, postData, axiosConfig)
          .then(response => {
            console.log('Sent specific features per model!')
             this.getFinalResults()
          })
          .catch(error => {
            console.log(error)
          })
    },
    UpdateBrushBoxPlot () {
      EventBus.$emit('emittedEventCallingBrushedBoxPlot', this.brushedBoxPlotUpdate)
    },
    CallPCP () {
      EventBus.$emit('emittedEventCallingSelectedALgorithm', this.selectedAlgorithm)
      EventBus.$emit('emittedEventCallingModelClear')
      EventBus.$emit('emittedEventCallingModelSelect', this.selectedAlgorithm)
      EventBus.$emit('emittedEventCallingModel', this.PerformancePerModel)
    },
    Reset () {
      const path = `http://127.0.0.1:5000/data/Reset`
      this.reset = true
      this.firstTimeExec = true
      this.OverSelLength = 0
      this.OverAllLength = 0
      this.OverSelLengthCM = 0
      this.OverAllLengthCM = 0
      this.sankeyCallS = 1
      this.mode = 0
      this.CurrentStage = 1
      const postData = {
        ClassifiersList: this.reset
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('The server side was reset! Done.')
          this.reset = false
          EventBus.$emit('resetViews')
        })
        .catch(error => {
          console.log(error)
        })
    },
    render (flag) {
      this.combineWH = []
      this.width = document.body.clientWidth / 12 - 30
      this.height = document.body.clientHeight / 3
      this.combineWH.push(this.width)
      this.combineWH.push(this.height)
      if(flag) {
        EventBus.$emit('Responsive', this.combineWH)
      }
      else {
        EventBus.$emit('ResponsiveandChange', this.combineWH)
      }
    },
    change () {
      this.render(false)
    },
    factors () {
      const path = `http://127.0.0.1:5000/data/factors`
      const postData = {
        Factors: this.basicValuesFact
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('Sending factors!')
          this.getDatafromtheBackEnd()
        })
        .catch(error => {
          console.log(error)
        })
    },
    RetrieveNewColors () {
      const path = `http://127.0.0.1:5000/data/UpdateOverv`

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.get(path, axiosConfig)
        .then(response => {
          this.sumPerClassifierSel = response.data.Results
          console.log('Server successfully send the new colors!')
          EventBus.$emit('getColors',this.sumPerClassifierSel)
        })
        .catch(error => {
          console.log(error)
        })
    },
    updateToggle () {
      var toggles = []
      toggles.push(this.toggle1)
      toggles.push(this.toggle2)
      toggles.push(this.toggle3)
      EventBus.$emit('emittedEventCallingTogglesUpdate', toggles)
    },
    DataSpaceFun () {
      const path = `http://127.0.0.1:5000/data/SendDataSpacPoints`
      const postData = {
        points: this.dataPointsSelfromDataSpace,
      }
      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('Send request to server! Brushed points sent successfully!')
        })
        .catch(error => {
          console.log(error)
        })
    },
    swapSymbol () {
      var off = document.getElementById('mode0');
      var on = document.getElementById('mode1');
      EventBus.$emit('ON', true)
      on.style.display = 'inline'
      off.style.display = 'none'
    },
    sendPointsCrossMutat () {
      const path = `http://127.0.0.1:5000/data/CrossoverMutation`
      for (let i = 0; i < this.storeEnsemble.length; i++) {
        this.storeEnsemblePermanently.push(this.storeEnsemble[i])
      }
      var mergedStoreEnsembleLoc = [].concat.apply([], this.storeEnsemblePermanently)

      if (this.CurrentStage == 1) {
        var postData = {
          RemainingPoints: this.unselectedRemainingPoints,
          StoreEnsemble: mergedStoreEnsembleLoc,
          loopNumber: this.CMNumberofModelsOFFICIAL,
          Stage: this.CurrentStage
        }
      } else {
        var postData = {
          RemainingPoints: this.unselectedRemainingPoints,
          StoreEnsemble: mergedStoreEnsembleLoc,
          loopNumber: this.CMNumberofModelsOFFICIALS2,
          Stage: this.CurrentStage
        }
        this.sankeyCallS = this.sankeyCallS + 1
      }

      const axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token',
          'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS'
        }
      }
      axios.post(path, postData, axiosConfig)
        .then(response => {
          console.log('Sent the unselected points for crossover and mutation.')
          this.CurrentStage = 2
          this.getDatafromtheBackEnd()
          this.getCMComputedData()
          this.changeActiveTo2()
        })
        .catch(error => {
          console.log(error)
        })
    },
    changeActiveTo1 () {
      this.projectionID_A = 1 
      this.projectionID_B = 1
      EventBus.$emit('activeNow', 1)
    },
    changeActiveTo2 () {
      this.projectionID_A = 2 
      this.projectionID_B = 2
      EventBus.$emit('activeNow', 2)
    }
  },
  created () {

    // does the browser support the Navigation Timing API?
    if (window.performance) {
        console.info("window.performance is supported");
    }
    // do something based on the navigation type...
    if(performance.navigation.type === 1) {
        console.info("TYPE_RELOAD");
        this.Reset();
    }
    window.addEventListener('resize', this.change)
  },
  mounted() {

    var modal = document.getElementById('myModal')
    window.onclick = function(event) {
      //alert(event.target)
        if (event.target == modal) {
            modal.style.display = "none";
        } 
    }
    this.render(true)
    loadProgressBar()
    window.onbeforeunload = function(e) {
      return 'Dialog text here.'
    }
    $(window).on("unload", function(e) {
      alert('Handler for .unload() called.');
    })



    EventBus.$on('sendKeyNow', data => { this.keyLoc = data })

    EventBus.$on('ReturningBrushedPointsIDs',  data => { this.modelsUpdate = data })
    //EventBus.$on('ReturningBrushedPointsIDs',  this.UpdateBarChartFeatures )
    EventBus.$on('ConfirmDataSet', this.fileNameSend)
    EventBus.$on('reset', this.changeActiveTo1)
    EventBus.$on('reset', this.Reset)
    EventBus.$on('ReturningAlgorithms', data => { this.selectedAlgorithms = data })
    EventBus.$on('ReturningBrushedPointsParams', data => { this.parametersofModels = data; })

    EventBus.$on('RemainingPoints', this.changeActiveTo1)
    EventBus.$on('RemainingPoints', data => { this.unselectedRemainingPoints = data })

    EventBus.$on('ChangeKey', data => { this.CurrentStage = data })
    EventBus.$on('InitializeCrossoverMutation', this.sendPointsCrossMutat)

    EventBus.$on('RemainingPointsCM', this.changeActiveTo2)
    EventBus.$on('RemainingPointsCM', data => { this.unselectedRemainingPointsEnsem = data })

    EventBus.$on('ChangeKey', data => { this.keyNow = data })
    EventBus.$on('SendSelectedPointsUpdateIndicator', data => { this.ClassifierIDsList = data; this.storeEnsemble = []; this.storeEnsemble.push(this.ClassifierIDsList) })
    EventBus.$on('SendSelectedPointsUpdateIndicator', this.SelectedPoints)
    EventBus.$on('sendToServerSelectedScatter', this.SendSelectedPointsToServer)

    EventBus.$on('SendSelectedPointsUpdateIndicatorCM', data => { this.ClassifierIDsListCM = data; this.storeEnsemble = []; this.storeEnsemble.push(this.ClassifierIDsListCM) })
    EventBus.$on('SendSelectedPointsUpdateIndicatorCM', this.SelectedPointsCM)

    EventBus.$on('SendSelectedDataPointsToServerEvent', data => { this.DataPointsSel = data })
    EventBus.$on('SendSelectedDataPointsToServerEvent', this.SendSelectedDataPointsToServer)
    EventBus.$on('SendSelectedFeaturesEvent', data => { this.SelectedFeaturesPerClassifier = data })
    EventBus.$on('sendToServerFeatures', this.UpdateBasedonFeatures)
    EventBus.$on('SendToServerDataSetConfirmation', data => { this.RetrieveValueFile = data })
    EventBus.$on('SendToServerLocalFile', data => { this.localFile = data })
    EventBus.$on('SendToServerLocalFile', this.SendToServerData)
    EventBus.$on('PCPCall', data => { this.selectedAlgorithm = data })
    EventBus.$on('toggle1', data => { this.toggle1 = data })
    EventBus.$on('toggle2', data => { this.toggle2 = data })
    EventBus.$on('toggle3', data => { this.toggle3 = data })
    EventBus.$on('toggle1', this.updateToggle)
    EventBus.$on('toggle2', this.updateToggle)
    EventBus.$on('toggle3', this.updateToggle)
    EventBus.$on('PCPCall', this.CallPCP)
    EventBus.$on('PCPCallDB', this.SendBrushedParameters)
    EventBus.$on('UpdateBoxPlot', data => { this.brushedBoxPlotUpdate = data })
    EventBus.$on('UpdateBoxPlot', this.UpdateBrushBoxPlot)
    EventBus.$on('CallFactorsView', data => { this.basicValuesFact = data })
    EventBus.$on('CallFactorsView', this.factors)
    EventBus.$on('AllAlModels', data => {
      this.valueSel = data
      this.valueAll = data
    })
    EventBus.$on('sendPointsNumber', data => {this.OverSelLength = data})
    EventBus.$on('sendPointsNumber', data => {this.OverAllLength = data})
    EventBus.$on('sendPointsNumberCM', data => {this.OverSelLengthCM = data})
    EventBus.$on('sendPointsNumberCM', data => {this.OverAllLengthCM = data})
  
    EventBus.$on('AllSelModels', data => {this.valueSel = data})

    EventBus.$on('RemoveFromEnsemble', data => { 
      this.ClassifierIDsListRemaining = data;
     })
    EventBus.$on('RemoveFromEnsemble', this.RemoveFromEnsembleModels)

    EventBus.$on('OpenModal', this.openModalFun)

    EventBus.$on('SendSelectedPointsToServerEventfromData', data => {this.dataPointsSelfromDataSpace = data})
    EventBus.$on('SendSelectedPointsToServerEventfromData', this.DataSpaceFun)

    EventBus.$on('SendFilter', data => {this.filterData = data})
    EventBus.$on('SendFilter', this.FilterFun)

    EventBus.$on('SendAction', data => {this.actionData = data})
    EventBus.$on('SendAction', this.ActionFun)

    EventBus.$on('SendProvenance', data => {this.provenanceData = data})
    EventBus.$on('SendProvenance', this.ProvenanceControlFun)

    EventBus.$on('toggleDeep', data => {this.toggleDeepMain = data})

    EventBus.$on('SendtheChangeinRangePos', data => { this.RandomSear = data })
    EventBus.$on('SendtheChangeinRangeNeg', data => { this.crossVal = data })
    EventBus.$on('factorsChanged', data => { this.basicValuesFact = data })

    EventBus.$on('changeValues', data => { this.CMNumberofModelsOFFICIAL = data })
    EventBus.$on('changeValues2Run', data => { this.CMNumberofModelsOFFICIALS2 = data })

    //Prevent double click to search for a word. 
    document.addEventListener('mousedown', function (event) {
      if (event.detail > 1) {
      event.preventDefault();
      }
    }, false);
  },
})
</script>

<style lang="scss">

#nprogress .bar {
background: red !important;
}

#nprogress .peg {
box-shadow: 0 0 10px red, 0 0 5px red !important;
}

#nprogress .spinner-icon {
border-top-color: red !important;
border-left-color: red !important;
}

body {
  font-family: 'Helvetica', 'Arial', sans-serif !important;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  margin-top: -4px !important;
  overflow-x: hidden !important;
}

.modal-backdrop {
  z-index: -1 !important;
}

.card-body {
   padding: 0.60rem !important;
}

hr {
  margin-top: 1rem;
  margin-bottom: 1rem;
  border: 0;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

@import './../assets/w3.css';
</style>