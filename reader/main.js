const { app , ipcRenderer , BrowserWindow } = require('electron')
let tesseract = require('tesseract.js')
const path = require('path')

app.on('ready',() => {
    let main = new BrowserWindow({
        width : 1000,
        height: 500,
    })

    main.loadFile(`${__dirname}/app/index.html`)


    console.log(__dirname);
    

    // tesseract.create({
    //     langPath : `${__dirname}/app/langs/`,
    //     workerPath : path.join(__dirname,'/node_modules/tesseract.js/src/node/worker.js'),
    //     corePath : path.join(__dirname,'/node_modules/tesseract.js/src/index.js'),
    // }).recognize(`${__dirname}/app/img/stroke-2.jpg`,{
    //     lang : 'eng',
    //     tessedit_char_whitelist: '0123456789'
    // }).progress((result) => {
    //     console.log(result)
    // }).then((content) => {
    //     console.log(content.text)
    // })
    
    main.on('closed', () => {
        console.log('closed')
        main = null
    })

})