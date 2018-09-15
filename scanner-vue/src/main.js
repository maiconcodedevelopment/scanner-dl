// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from "vue";
import App from "./App";
import router from "./router";

import { library } from "@fortawesome/fontawesome-svg-core";
import {
  faCoffee,
  faBarcode,
  faCodeBranch,
  faSearch,
  faFolder,
  faFolderOpen,
  faImage,
  faImages,
  faFileImage,
  faTrash,
  faTrashAlt,
  faTimes
} from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";

library.add(faCoffee);
library.add(faBarcode);
library.add(faCodeBranch);
library.add(faSearch);
library.add(faFolder);
library.add(faFolderOpen);
library.add(faImage);
library.add(faImages);
library.add(faFileImage);
library.add(faTrash);
library.add(faTrashAlt);
library.add(faTimes);

Vue.component("font-awesome-icon", FontAwesomeIcon);

Vue.config.productionTip = false;

/* eslint-disable no-new */
new Vue({
  el: "#app",
  router,
  components: { App },
  template: "<App/>"
});
