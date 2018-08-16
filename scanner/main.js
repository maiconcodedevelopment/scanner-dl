const { app , BrowserWindow , ipcRenderer } = require('electron')


app.on('ready',() => {

    let main = new BrowserWindow({
        width : 400,
        height : 400
    })

    main.loadFile(`${__dirname}/app/index.html`)

    main.on('closed',() => {
        main = null
    })
    // main.setBounds({
    //     x : 
    // })
})

app.isReady((params) => {
    console.log(params)
})


app.on('window-all-closed',() =>{
   app.quit()
})



