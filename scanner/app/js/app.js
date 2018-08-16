const { app , WebRequest , ipcMain , net , nativeImage } = require('electron')
const Tesseract = require('tesseract.js')


let config = {
   tessedit_char_whitelist : '0123456789'   
}

let img = nativeImage.createFromPath(`${__dirname}/img/scanner.png`)

Tesseract.create({
    langPath : `/opt/lampp/htdocs/scanner/app/langs/eng.traineddata`
}).recognize(`/opt/lampp/htdocs/scanner/app/img/scanner.png`,{
    lang : 'eng'
}).progress(function(process) {
    console.log(process)
}).then(function(result){
    console.log(result)
})
    
