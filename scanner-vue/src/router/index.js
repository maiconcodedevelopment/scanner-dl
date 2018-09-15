import Vue from "vue";
import Router from "vue-router";

import home from "@/scenes/home/index";
import process from "@/scenes/home/scenes/process/index";

//components
import selectpath from "@/components/path/select-path";

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: "/",
      name: "home",
      component: home,
      children: [
        {
          path: "/",
          name: "selectPath",
          component: selectpath
        },
        {
          path: "process",
          name: "process",
          component: process
        }
      ]
    }
  ]
});
