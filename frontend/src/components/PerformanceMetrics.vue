<template>
<div>
<b-row>
    <b-col cols="12">
      <table class="table table-borderless table-sm">
        <tbody>
          <tr>
            <th scope="col" colspan="2">Balanced data set</th>
            <th scope="col" colspan="2">Imbalanced data set</th>
          </tr>
          <tr>
            <td>(M1) Accuracy:</td>
            <td>
              <b-form-checkbox
                id="checkboxAcc"
                v-model="checkedAcc"
                @click="clickAcc"
              >
              </b-form-checkbox>
            </td>
            <td>(M1*) G-mean:</td>
            <td>
              <b-form-checkbox
                id="checkboxGM"
                v-model="checkedGM"
                @click="clickGM"
              >
              </b-form-checkbox>
            </td>
          </tr>
          <tr>
            <td>(M2) Precision:</td>
            <td>
              <b-form-checkbox
                id="checkboxPrec"
                v-model="checkedPrec"
                @click="clickPrec"
              >
              </b-form-checkbox>
            </td>
            <td>(M2*) ROC AUC:</td>
            <td>
              <b-form-checkbox
                id="checkboxRA"
                v-model="checkedRA"
                @click="clickRA"
              >
              </b-form-checkbox>
            </td>
          </tr>
          <tr>
            <td>(M3) Recall:</td>
            <td>
              <b-form-checkbox
                id="checkboxRec"
                v-model="checkedRec"
                @click="clickRec"
              >
              </b-form-checkbox>
            </td>
            <td>(M3*) Log loss:</td>
            <td>
              <b-form-checkbox
                id="checkboxLog"
                v-model="checkedLog"
                @click="clickLog"
              >
              </b-form-checkbox>
            </td>
          </tr>
          <tr>
            <td>(M4) F1-score:</td>
            <td>
              <b-form-checkbox
                id="checkboxF1"
                v-model="checkedF1"
                @click="clickF1"
              >
              </b-form-checkbox>
            </td>
            <td>(M4*) MCC:</td>
            <td>
              <b-form-checkbox
                id="checkboxMCC"
                v-model="checkedMCC"
                @click="clickMCC"
              >
              </b-form-checkbox>
            </td>
          </tr>
        </tbody>
      </table>
    </b-col>
</b-row>
</div>
</template>

<script>
    import { EventBus } from '../main.js'

    export default {
        name: 'PerformanceMetrics',
        data () {
          return {
              checkedAcc: true,
              checkedPrec: true,
              checkedRec: true,
              checkedF1: true,
              checkedGM: false,
              checkedRA: false,
              checkedLog: false,
              checkedMCC: false,
          }
        },
        methods: {
          clickAcc () {
            this.checkedAcc = !this.checkedAcc
          },
          clickPrec () {
            this.checkedPrec = !this.checkedPrec
          },
          clickRec () {
            this.checkedRec = !this.checkedRec  
          },
          clickF1 () {
            this.checkedF1 = !this.checkedF1 
          },
          clickGM () {
            this.checkedGM = !this.checkedGM
          },
          clickRA () {
            this.checkedRA = !this.checkedRA
          },
          clickLog () {
            this.checkedLog = !this.checkedLog
          },
          clickMCC () {
            this.checkedMCC = !this.checkedMCC
          },
        },
        mounted () {

        }
    }
</script>

<style>
</style>