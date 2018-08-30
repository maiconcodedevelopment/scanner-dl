<template>
    <div class="scanner__list">
       <div class="scanner__list__container">
           <div @click="scannerOn" class="scanner__list__card">
               <div class="scanner__list__card__done">
                   <font-awesome-icon icon="check" />
               </div>
               <div class="scanner__list__card__barcode">
                   <font-awesome-icon icon="barcode" />
               </div>
               <h5>Scaneando Renavam</h5>
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
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-direction: row;
    height: 65px;

    overflow: hidden;
    border-top-left-radius: 2px;
    border-bottom-left-radius: 2px;
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
        
    }
}
</style>
