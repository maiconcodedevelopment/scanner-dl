<template>
   <div class="select-path">
       <img class="select-path-icone" v-bind:src="icones.path" />
       <h2>Selecione uma pasta de origem</h2>
       <p>Escolha uma pasta no seu computador onde contém todos os documentos que deseja escanear.</p>
       <button type="button" v-on:click="selectFolder" > <font-awesome-icon icon="search" />   Procurar no computador</button>
   </div>
</template>

<script>
const { ipcRenderer } = window.require("electron");
import { icones } from "@/images";

export default {
  name: "select-path",
  data() {
    return {
      icones
    };
  },
  methods: {
    selectFolder() {
      ipcRenderer.send("select-path", "teste do argumento");
      this.$router.push({ name: "process" });
      console.log("push name");
    }
  }
};
</script>


<style lang="scss" scoped>
.select-path {
  width: 572px;
  height: 361px;
  border-radius: 3px;
  background-color: #ffffff;

  margin: 0 auto;
  padding: 0px 95px;

  display: flex;
  justify-items: center;
  justify-content: center;
  flex-direction: column;
  align-items: center;

  align-self: center;
  justify-self: center;

  .select-path-icone {
  }

  h2 {
    font-family: Roboto;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    color: #565655;
    margin-top: 26px;
  }
  p {
    margin-top: 11px;
    font-family: Roboto;
    font-size: 14px;
    font-weight: normal;
    text-align: center;
    color: #7a7a79;
    margin-bottom: 46px;
  }
  button {
    width: 222px;
    height: 54px;
    border-radius: 3px;
    background-color: #ffef14;
    border-style: none;

    cursor: pointer;
  }
}
</style>
