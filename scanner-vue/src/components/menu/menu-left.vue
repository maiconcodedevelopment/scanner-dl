<template>
   <div class="menu-left">
       <div class="menu-left-logo">
           <img v-bind:src="logos.helpay" />
       </div>
       <ul class="menu-left-list">
           <li>
                <font-awesome-icon icon="barcode" />
                <p>Scanner do Renavam</p>
           </li>
       </ul>
   </div>
</template>


<script>
import { logos } from "@/images"

export default {
    name : "menu-left",
    data(){
        return {
            logos
        }
    }
}
</script>


<style lang="scss" scoped>
.menu-left{
    height: 100vh;
    width: 120px;
    background-color: #565655;
    .menu-left-logo{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        
        height: 120px;
        width: 100%;
        border-bottom : 1px solid #7a7a79;
    }
    .menu-left-list{
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        li{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
    
            width: 100%;
            height: 112px;
            padding: 10px;
            font-size: 20px;
    
            border-bottom : 1px solid #7a7a79;
            p{
               font-family: Roboto;
               font-size: 12px;
               font-weight: 500;
               font-style: normal;
               line-height: 1.17;
               text-align: center;
               color: #adadab;
            }
        }
    }
}
</style>
