<template>
    <div id="container__scanner">
        <menubarleft></menubarleft>
        <scannerlist></scannerlist>
    </div>
</template>


<script>
import menubarleft from "@/components/menu"
import scannerlist from "@/components/scanner/scanner-list";

export default {
    name : "main",
    components : {
        menubarleft,
        scannerlist
    }
}
</script>


<style lang="scss" scoped>
#container__scanner{
    width: 100vw;
    height: 100vh;
    display: flex;
    align-items: flex-start;
    justify-content: flex-start;
    flex-direction: row;
}
</style>
