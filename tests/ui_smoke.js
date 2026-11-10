// Loads the finished page in jsdom, clicks the demo button and checks that results render.
const { JSDOM } = require("jsdom");
const fs = require("fs");
const html = fs.readFileSync("web/index.html", "utf8");
const errors = [];
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true,
  virtualConsole: new (require("jsdom").VirtualConsole)().on("jsdomError", e => { if (!/getContext/.test(e.message)) errors.push(e.message) }) });
const d = dom.window.document;
(async () => {
  const assert = require("assert");
  assert(d.getElementById("mc").textContent.includes("Random forest baseline"), "model card rendered");
  d.getElementById("demo").click();
  for (let i = 0; i < 100 && d.getElementById("res").hidden; i++) await new Promise(r => setTimeout(r, 200));
  assert(!d.getElementById("res").hidden, "results shown: " + d.getElementById("warn").textContent + errors.join(";"));
  const rows = d.querySelectorAll("#tb tr").length, cards = d.querySelectorAll("#cards .c").length;
  assert(rows >= 6 && cards === 6, `rows ${rows} cards ${cards}`);
  console.log("table:\n" + [...d.querySelectorAll("#tb tr")].map(r => [...r.children].map(c => c.textContent).join(" | ")).join("\n"));
  console.log("cards:", [...d.querySelectorAll("#cards .c")].map(c => c.textContent).join(" ; "));
  console.log("warnings:", d.getElementById("warn").textContent || "none");
  assert(errors.length === 0, "page errors: " + errors.join(";"));
  console.log("UI smoke test passed");
  process.exit(0);
})().catch(e => { console.error("FAIL", e.message); process.exit(1) });
