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
});

// session.defaultSession.se

ipcMain.on("select-path", (event, args) => {
  console.log(args);
  dialog.showOpenDialog(
    { properties: ["openDirectory", "multiSelections"] },
    dirPath => {
      let options = {
        mode: "text",
        pythonPath: "/usr/bin/python3",
        pythonOptions: ["-u"], // get print results in real-time
        scriptPath: "./",
        args: ["value1", "value2", "value3"]
      };

      pytonShell.PythonShell.run(
        `${__dirname}/robot.py`,
        options,
        (err, results) => {
          if (err) throw err;
          console.log(results);
        }
      );
    }
  );
});

app.on("window-all-closed", () => {
  app.quit();
});
