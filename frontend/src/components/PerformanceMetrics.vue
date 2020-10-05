<template>
<div>
<b-row>
    <b-col cols="12">
      <table class="table table-borderless table-sm">
        <tbody>
          <tr>
            <th scope="col" colspan="1">Balanced data set</th>
            <td>
            <b-form-checkbox
                id="checkedboxBalanced"
                v-model="checkedBalanced"
                @change="clickBalan()"
              >
            </b-form-checkbox>
            </td>
            <th scope="col" colspan="1">Imbalanced data set</th>
            <td>
            <b-form-checkbox
                id="checkedboxUnbalanced"
                v-model="checkedUnbalanced"
                @change="clickUnbalan()"
              >
            </b-form-checkbox>
            </td>
          </tr>
          <tr>
            <td>(M1) Accuracy:</td>
            <td>
              <b-form-checkbox
                id="checkboxAcc"
                v-model="checkedAcc"
                @change="clickAcc()"
              >
              </b-form-checkbox>
            </td>
            <td>(M1*) G-mean:</td>
            <td>
              <b-form-checkbox
                id="checkboxGM"
                v-model="checkedGM"
                @change="clickGM()"
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
                @change="clickPrec()"
              >
              </b-form-checkbox>
            </td>
            <td>(M2*) ROC AUC:</td>
            <td>
              <b-form-checkbox
                id="checkboxRA"
                v-model="checkedRA"
                @change="clickRA()"
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
                @change="clickRec()"
              >
              </b-form-checkbox>
            </td>
            <td>(M3*) Log loss:</td>
            <td>
              <b-form-checkbox
                id="checkboxLog"
                v-model="checkedLog"
                @change="clickLog()"
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
                @change="clickF1()"
              >
              </b-form-checkbox>
            </td>
            <td>(M4*) MCC:</td>
            <td>
              <b-form-checkbox
                id="checkboxMCC"
                v-model="checkedMCC"
                @change="clickMCC()"
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
              checkedBalanced: true,
              checkedUnbalanced: false,
              factorsLocal: [1,1,1,1,0,0,0,0]
          }
        },
        methods: {
          clickBalan () {
            this.checkedUnbalanced = !this.checkedUnbalanced
            this.checkedBalanced = !this.checkedBalanced

            this.checkedAcc = !this.checkedAcc
            this.checkedPrec = !this.checkedPrec
            this.checkedF1 = !this.checkedF1
            this.checkedGM = !this.checkedGM
            this.checkedRA = !this.checkedRA
            this.checkedMCC = !this.checkedMCC
            this.checkedLog = !this.checkedLog
            this.checkedRec = !this.checkedRec

            this.factorsLocal = [1,1,1,1,0,0,0,0]
            EventBus.$emit('factorsChanged', this.factorsLocal)
          },
          clickUnbalan () {
            this.checkedUnbalanced = !this.checkedUnbalanced
            this.checkedBalanced = !this.checkedBalanced

            this.checkedAcc = !this.checkedAcc
            this.checkedPrec = !this.checkedPrec
            this.checkedF1 = !this.checkedF1
            this.checkedGM = !this.checkedGM
            this.checkedRA = !this.checkedRA
            this.checkedMCC = !this.checkedMCC
            this.checkedLog = !this.checkedLog
            this.checkedRec = !this.checkedRec

            this.factorsLocal = [0,0,0,0,1,1,1,1]
            EventBus.$emit('factorsChanged', this.factorsLocal)
          },
          clickAcc () {
            this.checkedAcc = !this.checkedAcc
            this.factorsRegisterChange(0,this.checkedAcc)
          },
          clickPrec () {
            this.checkedPrec = !this.checkedPrec
            this.factorsRegisterChange(1,this.checkedPrec)
          },
          clickRec () {
            this.checkedRec = !this.checkedRec
            this.factorsRegisterChange(2,this.checkedRec)
          },
          clickF1 () {
            this.checkedF1 = !this.checkedF1
            this.factorsRegisterChange(3,this.checkedF1)
          },
          clickGM () {
            this.checkedGM = !this.checkedGM
            this.factorsRegisterChange(4,this.checkedGM)
          },
          clickRA () {
            this.checkedRA = !this.checkedRA
            this.factorsRegisterChange(5,this.checkedRA)
          },
          clickLog () {
            this.checkedLog = !this.checkedLog
            this.factorsRegisterChange(6,this.checkedLog)
          },
          clickMCC () {
            this.checkedMCC = !this.checkedMCC
            this.factorsRegisterChange(7,this.checkedMCC)
          },
          factorsRegisterChange (position,value) {
            if (value == true) {
              this.factorsLocal[position] = 1
            } else {
              this.factorsLocal[position] = 0
            }
            EventBus.$emit('factorsChanged', this.factorsLocal)
          },
          resetBoxes () {
            this.checkedAcc = true
            this.checkedPrec = true
            this.checkedRec = true
            this.checkedF1 = true
            this.checkedGM = false
            this.checkedRA = false
            this.checkedLog = false
            this.checkedMCC = false

            this.factorsLocal = [1,1,1,1,0,0,0,0]
            EventBus.$emit('factorsChanged', this.factorsLocal)
          }
        },
        mounted () {
          EventBus.$on('reset', this.resetBoxes)
        }
    }
</script>

<style>
</style>