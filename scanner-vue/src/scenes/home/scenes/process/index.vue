<template>
    <div class="container__process">

          <div class="container__process__content">
            <div class="container__process__head">
                <font-awesome-icon icon="folder-open"/>
                <div class="head__path">
                    <h4 class="head__path__title">Pasta de origem</h4>
                    <p class="head__path__adress"></p>
                    <button type="button" @click="selectFolder">Alterar</button>
                </div>
            </div>
            <div class="container__process__document">
                <div class="process__document__head">
                    <font-awesome-icon icon="image" /> 
                    <div class="document__head">
                        <h4 class="document__head__title">Arquivos encontrados</h4>
                        <p class="document__head__subtitle">Confira a relação das imagens encontradas na pasta selecionada</p>
                    </div>
                </div>
                <div class="process__document__list">
                    <div class="process__document__list__head">
                        <div class="process__document__select" >
                            <input type="checkbox" name="checkbox__all" id="checkbox__all" class="process__document__checkbox" :class="{ listactiveall }" >
                            <label for="checkbox__all"  @click="activelisteAll" class="checkmark"></label>
                        </div>
                        <h4 class="process__document__title">NOME DO ARQUIVO</h4>
                    </div>
                    <div class="process__document__list__cards">
                      <cardscanner v-for="(scanner,index) in listscanner" 
                                   v-bind:key="index" 
                                   :path="scanner.path"
                                   :active="scanner.active"
                                   @remove="remove"
                                   :scanner="scanner"
                                   v-model="scanner.active" />
                    </div>
                </div>
            </div>
            <div class="process__document__start" v-show="showbarbottom">
                  <h5 class="process__document_start__title">Você selecionou 20 arquivos</h5>
                  <button type="button" @click="scannerDocuments" > <font-awesome-icon icon="barcode"/> Escanear arquivos selecionados</button>
                </div>
          </div>
          <div class="container__process__document__scanner">
              <scannerdocument/>
          </div>

    </div>
</template>


<script>
const { ipcRenderer } = window.require("electron");

import { icones } from "@/images";

import scannerdocument from "@/components/scanner/scanner-document";
import cardscanner from "@/components/listcard/card-scanner";

export default {
  name: "scanner",
  components: {
    cardscanner,
    scannerdocument
  },
  data() {
    return {
      listscanner: [],
      listactive: false,
      listactiveall: false,
      showbarbottom: true
    };
  },
  mounted() {
    ipcRenderer.on("paths", (event, list) => {
      list = list.replace(/'/g, '"');
      list = JSON.parse(list);
      list.map((path, index) => {
        this.listscanner.push({ path, active: false });
      });
    });
  },
  watch: {
    listactiveall: function(value, oldvalue) {
      if (value) {
        this.listscanner.map(scanner => (scanner.active = true));
      } else {
        this.listscanner.map(scanner => (scanner.active = false));
      }
    }
  },
  methods: {
    selectFolder() {
      this.showbarbottom = true;
      this.listactiveall = false;
      this.listactive = false;

      ipcRenderer.send("select-path", "teste do argumento");
      this.$router.push({ name: "process" });
      console.log("push name");
    },
    activelisteAll() {
      this.listactiveall = !this.listactiveall;
      console.log("sim");
    },
    remove(path) {
      this.listscanner.map((scanner, index) => {
        if (scanner.path == path) {
          this.listscanner.splice(index, 1);
        }
      });
    },
    scannerDocuments() {
      let documents = this.listscanner.filter(scanner => {
        return scanner.active;
      });

      if (documents.length > 0) {
        this.showbarbottom = false;
        this.scannerDocumentsRenavam();
      } else {
        console.log("not length");
      }
    },
    scannerDocumentsRenavam() {
      console.log("aqui");
      let paths = [];
      this.listscanner.map(scanner => {
        paths.push(scanner.path);
      });
      ipcRenderer.send("scanner-document", paths);
    }
  }
};
</script>

<style lang="scss" scoped>
.container__process {
  width: 100%;

  display: flex;
  align-items: center;
  justify-content: flex-start;
  flex-direction: row;

  .container__process__content {
    display: flex;
    align-items: center;
    flex-direction: column;
    justify-content: flex-start;
    flex: 1;

    height: 100%;
  }

  .container__process__document__scanner {
    width: 512px;
    height: 100%;
    background-color: white;
  }

  .container__process__head {
    height: 120px;
    border: solid 1px #e2e2e2;
    width: 100%;

    display: flex;
    align-items: flex-start;
    justify-content: flex-start;

    padding: 46px 10% 0px 10%;

    .head__path {
      height: 40px;
      margin-left: 10px;
      position: relative;

      .head__path__title {
        font-family: Roboto;
        font-size: 16px;
        font-weight: bold;
        text-align: left;
        color: #565655;
        margin-top: -3px;
      }

      .head__path__adress {
        font-family: Roboto;
        font-size: 13px;
        font-weight: normal;
        text-align: left;
        color: #7a7a79;
      }

      button[type="button"] {
        position: absolute;
        right: -20px;
        bottom: 0;
        border-style: none;
        width: auto;
        height: auto;
        font-family: Roboto;
        font-size: 12px;
        font-weight: 500;
        text-align: center;
        color: #565655;
      }
    }
  }
  .container__process__document {
    position: relative;
    width: 100%;
    flex: 1;
    padding: 0px 10% 0px 10%;

    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    flex-direction: column;

    .process__document__head {
      width: 100%;
      height: 120px;

      padding: 40px 0px 0px 0px;

      display: flex;
      align-items: flex-start;
      justify-content: flex-start;
      flex-direction: row;

      .document__head {
        margin-top: -2px;
        margin-left: 10px;
        display: flex;
        align-items: flex-start;
        justify-content: flex-start;
        flex-direction: column;

        .document__head__title {
          font-family: Roboto;
          font-size: 16px;
          font-weight: bold;
          font-style: normal;
          letter-spacing: normal;
          text-align: left;
          color: #565655;
          margin-bottom: 3px;
        }
        .document__head__subtitle {
          font-family: Roboto;
          font-size: 13px;
          font-style: normal;
          text-align: left;
          color: #7a7a79;
        }
      }
    }
    .process__document__list {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;

      padding: 0px;
      width: 100%;
      height: 100%;
      .process__document__list__head {
        width: 100%;
        height: 65px;

        display: flex;
        align-items: center;
        justify-content: flex-start;
        flex-direction: row;

        background-color: #565655;
        padding: 0px 20px;
        .process__document__select {
          position: relative;
          width: 18px;
          height: 18px;

          .checkmark {
            width: 100%;
            height: 100%;
            position: absolute;
            left: 0;
            top: 0;
            border: 2px solid #e1e1e0;
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
              border: solid white;
              border-width: 0 2px 2px 0;
              -webkit-transform: rotate(45deg);
              -ms-transform: rotate(45deg);
              transform: rotate(45deg);
            }
          }
          .process__document__checkbox {
            position: absolute;
            opacity: 0;
            cursor: pointer;
            &.listactiveall + .checkmark {
              display: block;
              background-color: #f6e60a;
              border: 0;
              &::after {
                display: block;
              }
            }
          }
        }
        .process__document__title {
          margin-left: 10px;
          font-family: Roboto;
          font-size: 12px;
          font-weight: 500;
          text-align: left;
          color: #ffffff;
        }
      }
      .process__document__list__cards {
        flex: 1;
        width: 100%;
        height: 100%;
        overflow-y: scroll;
      }
    }
  }
  .process__document__start {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-direction: row;

    padding: 0px 10% 0px 10%;

    width: 100%;
    height: 95px;
    background-color: white;
    .process__document_start__title {
      font-family: Roboto;
      font-size: 16px;
      font-weight: bold;
      text-align: left;
      color: #565655;
    }
    button[type="button"] {
      width: 282px;
      height: 54px;
      border-radius: 3px;
      background-color: #ffef14;

      display: flex;
      align-items: center;
      justify-content: space-around;
      flex-direction: row;
      border-style: none;
      cursor: pointer;
      font-family: Roboto;
      font-size: 14.5px;
      font-weight: 500;
      color: #565655;
    }
  }
}
</style>
