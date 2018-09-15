<template>
    <div class="container">
        <menuleft></menuleft>
        <div class="container__content">
            <router-view></router-view>
            <!-- <selectpath></selectpath> -->
        </div>
    </div>
</template>


<script>
import menuleft from "@/components/menu/menu-left";
import selectpath from "@/components/path/select-path";

export default {
  name: "home",
  components: {
    menuleft,
    selectpath
  }
};
</script>

<style lang="scss" scoped>
.container {
  display: flex;
  flex-direction: row;
  justify-content: flex-start;
  align-items: center;
  .container__content {
    flex: 1;
    display: flex;
    height: 100vh;
  }
}
</style>
