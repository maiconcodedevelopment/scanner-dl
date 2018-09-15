import Vue from "vue"
import Vuex from "vuex"

Vue.use(Vuex)

import { storeScanner } from "@/store/scanner/storeScanner"


export const store = new Vuex.Store({
    modules : {
        storeScanner
    }
})