<template>
    <div class="menu__bar__left">
        <ul class="menu__bar__left__container">
            <div class="menu__bar__left__logo">
                <img v-bind:src="logo.helpay.yellow" />
            </div>
            <ul class="menu__bar__left__ul">
              <li class="menu__bar__left__li">
                <font-awesome-icon icon="print" />
              </li>     
            </ul>
        </ul>
    </div>
</template>


<script>
import { logo } from "@/assets/images";

export default {
    name : "menubarleft",
    data : () => {
        return {
            logo
        }
    }
}
</script>


<style lang="scss" scoped>
   .menu__bar__left{
       height: 100vh;
       width: 130px;
       background-color: #565655;
       display: flex;
       align-items: center;
       justify-content: flex-start;
       flex-direction: column;
       .menu__bar__left__container{
           padding: 10px 10px;
           .menu__bar__left__logo{
               img{
                   width: 100%;
               }
           }
       }
       .menu__bar__left__ul{
           display: flex;
           align-items: center;
           justify-content: center;
           flex-direction: column;
           .menu__bar__left__li{
               width: 100%;
               height: 50px;

               cursor: pointer;

               background-color: #363635;
               color: white;
               font-size: 25px;
               list-style: none;
               display: flex;
               align-items: center;
               justify-content: center;
               border-radius: 2px;
           }
       }
   }
</style>
