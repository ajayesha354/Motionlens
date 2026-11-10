// Runs the website's own pipeline (extracted from web/index.html) on a CSV and prints JSON.
const fs = require("fs");
const html = fs.readFileSync("web/index.html", "utf8");
const code = html.split("/*LOGIC-START*/")[1].split("/*LOGIC-END*/")[0];
const api = new Function(code + ";return{D,parseCSV,detect,pipeline,summarize,toCSV}")();
const p = api.parseCSV(fs.readFileSync(process.argv[2], "utf8"));
const m = api.detect(p.head);
const r = api.pipeline(p.cols, m, { accUnit: process.argv[5] || "auto", gyroUnit: "auto", fs: +(process.argv[4] || 0), useGyro: process.argv[3] !== "acc", smooth: process.argv[6] === "smooth" });
console.log(JSON.stringify({ nw: r.nw, model: r.model, warn: r.warn, T: r.model.startsWith("Accelerometer +") ? api.D.full.T : api.D.acc.T,
  lg: r.lg.map(a => Array.from(a)), ld: r.ld, sm: r.sm, raw: r.raw }));
