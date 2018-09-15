<template>
    <div class="card__scanner">
       <div class="card__scanner__select">
           <input type="checkbox" id="checkbox__select" name="scanner" class="select__scanner">
           <label for="checkbox__select" class="checkmark"></label>
       </div>
       <h4 class="card__scanner__title">scanner_pape.jpg</h4>
       <div class="card__scanner__remove">
           <font-awesome-icon icon="trash-alt" />
       </div>
    </div>    
</template>


<script>
export default {
  name: "card-scanner",
  props: {
    name: {
      type: String,
      require: false
    }
  }
};
</script>

<style lang="scss" scoped>
.card__scanner {
  width: 100%;
  height: 62px;
  padding: 0px 20px;

  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-direction: row;

  background-color: white;
  border-radius: 3px;

  margin-top: 3px;

  .card__scanner__select {
    position: relative;
    width: 18px;
    height: 18px;

    .checkmark {
      position: absolute;
      width: 100%;
      height: 100%;
      top: 0;
      left: 0;
      border: 2px solid #adadab;
      border-radius: 3px;
      cursor: pointer;

      &::after {
        content: " ";
        position: absolute;
        display: none;
        left: 5px;
        top: 1px;
        width: 5px;
        height: 10px;
        height: 10px;
        border: solid white;
        border-width: 0 2px 2px 0;
        -webkit-transform: rotate(45deg);
        -ms-transform: rotate(45deg);
        transform: rotate(45deg);
      }
    }
    .select__scanner {
      position: absolute;
      opacity: 0;
      cursor: pointer;
      &:checked + .checkmark {
        border: none;
        background-color: #f6e60a;
        &::after {
          display: block;
        }
      }
    }
  }
  .card__scanner__title {
    margin-right: auto;
    margin-left: 10px;
    font-family: Roboto;
    font-size: 13px;
    font-weight: 500;
    color: #7a7a79;
  }
  .card__scanner__remove {
    cursor: pointer;
  }
}
</style>
