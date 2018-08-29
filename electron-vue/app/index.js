const {  BrowserWindow ,    app } = require('electron')
const electron = require('electron')


let win = null

app.on('ready',() => {
    win = new BrowserWindow({
       width : 1200,
       height : 800
    })

    win.loadURL("http://localhost:8080")

    win.on('closed',() => {
        win = null
    })
})


app.on('window-all-closed',() => {
    app.quit()
})