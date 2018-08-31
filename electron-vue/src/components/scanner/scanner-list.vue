<template>
    <div class="scanner__list">
       <div class="scanner__list__container">
           <div class="scanner__list__card" v-for="item in [1,2,3,4,5,6,7,8]" :key="item">
               <div class="scanner__list__card__done">
                   <font-awesome-icon icon="check" />
               </div>
               <div class="scanner__list__card__barcode">
                   <font-awesome-icon icon="barcode" />
               </div>
               <h5 class="scanner__list__title" >Scaneando Renavam</h5>
               <div class="scanner__list__card__delete">
                   <font-awesome-icon icon="trash" />
               </div>
           </div>
       </div>
    </div>    
</template>

<script>

export default {
    name : "scannerlist",
    mounted(){
    },
    methods:{
    }
}
</script>

<style lang="scss" scoped>
.scanner__list{
    width: 100%;
    height: auto;
    padding: 10px;
}

.scanner__list__container{
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    flex-direction: column;
}

.scanner__list__card{
    width: 100%;
    margin-top: 10px;

    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-direction: row;
    height: 65px;

    overflow: hidden;
    border-top-left-radius: 2px;
    border-bottom-left-radius: 2px;
    
    cursor: pointer;

    .scanner__list__card__done{
        display: flex;
        align-items: center;
        justify-content: center;

        background-color: #29eb7f;
        align-self: stretch;
        width: 65px;
        color: white;
        font-size: 25px;
    }
    .scanner__list__card__barcode{
        display: flex;
        align-items: center;
        justify-content: center;

        width: 65px;
        align-self: stretch;
        color: #363635;
        font-size: 25px;
        margin-right: 10px;
    }
    .scanner__list__title{
        margin-right: auto;
    }
    .scanner__list__card__delete{
        width: 45px;
        border-radius: 50px;
        background-color: #ff4949;
        
        display: flex;
        align-items: center;
        justify-content: center;
        height: 45px;

        color: white;
        cursor: pointer;
    }
}
</style>
