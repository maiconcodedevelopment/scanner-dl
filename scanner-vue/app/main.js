const { BrowserWindow, app, ipcMain, session } = require("electron");
const { dialog } = require("electron");

var pytonShell = require("python-shell");

const fs = require("fs");
let win = null;

app.on("ready", () => {
  let win = new BrowserWindow({ width: 1600, height: 1000 });
  win.loadURL("http://localhost:8080");

  console.log(app.getPath("desktop"));

  win.on("closed", () => {
    win = null;
  });

  let ses = session.fromPartition("persist:name");
  console.log(ses.getUserAgent());

  session.defaultSession.cookies.set(
    {
      name: "foo",
      value: "bar",
      url: "http"
    },
    err => {
      if (err) {
        console.log("Error", err);
      }
    }
  );

  ipcMain.on("select-path", (event, args) => {
    // console.log(args);
    dialog.showOpenDialog(
      { properties: ["openDirectory", "multiSelections"] },
      dirPath => {
        let pathFiles = dirPath[0];

        pytonShell.PythonShell.run(
          `${__dirname}/lerning/Paths.py`,
          { pythonOptions: ["-u"], args: [pathFiles] },
          (err, results) => {
            if (err) throw err;
            // list = results[0].replace(/'/g, '"');
            // list = JSON.parse(list);
            // list.map(path => {
            //   console.log(path);
            // });
            win.webContents.send("paths", results[0]);
          }
        );
      }
    );
  });

  ipcMain.on("scanner-document", (event, args) => {
    pytonShell.PythonShell.run(
      `${__dirname}/lerning/Despachante.py`,
      {
        pythonOptions: ["-u"],
        args: [args]
      },
      (err, resutls) => {
        if (err) throw err;
        event.sender.send("predicts", resutls);
        console.log(resutls);
      }
    );
  });
});

// session.defaultSession.se

app.on("window-all-closed", () => {
  app.quit();
});
