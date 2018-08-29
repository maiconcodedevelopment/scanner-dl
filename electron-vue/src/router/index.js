import Vue from 'vue'
import Router from 'vue-router'

import Main from "@/scenes/main";
import HelloWorld from '@/components/HelloWorld'


Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'Main',
      component: Main
    }
  ]
})
