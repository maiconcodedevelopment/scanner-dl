<template>
   <div class="scanner__document">
       <div class="scanner__document__head">
         <div class="scanner__document__head__visualise">
           <h5 class="scanner__document__head__title">Visualização do arquivo escaneado</h5>
           <p class="scanner__document__head__subtitle">scanner_002.jpg</p>
         </div>
         <div class="scanner__document__head__remove">
           <font-awesome-icon icon="times" />
         </div>
       </div>
      <div class="scanner__document__content">
        <div class="scanner__document__visualive">
          <img v-bind:src="images.scanner" />
        </div>
        <div class="scanner__document__content__input">

        </div>
      </div>
   </div>    
</template>


<script>
import { images } from "@/images";

export default {
  name: "scanner-document",
  data() {
    return {
      images
    };
  }
};
</script>

<style lang="scss" scoped>
.scanner__document {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;

  height: auto;
  width: 100%;
  .scanner__document__head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-direction: row;

    width: 100%;
    height: 120px;
    border-bottom: solid 1px #e2e2e2;
    padding: 42px;

    .scanner__document__head__visualise {
      display: flex;
      justify-content: flex-start;
      align-items: flex-start;
      flex-direction: column;
      .scanner__document__head__title {
        font-family: Roboto;
        font-size: 16px;
        font-weight: bold;
        color: #565655;
        margin-bottom: 4px;
      }
      .scanner__document__head__subtitle {
        font-family: Roboto;
        font-size: 13px;
        font-weight: normal;
        text-align: left;
        color: #7a7a79;
      }
    }
    .scanner__document__head__remove {
    }
  }
  .scanner__document__content {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    flex-direction: column;

    padding: 34px 0px 0px 0px;

    .scanner__document__visualive {
      width: 442px;
      height: 335px;
      border-radius: 3px;
      border: solid 1px #e2e2e2;
      padding: 3px;

      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      img {
        max-width: 100%;
        max-height: 100%;
      }
    }
  }
}
</style>
