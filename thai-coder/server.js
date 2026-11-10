// server.mjs — ระบบความจำ AI (Node.js) เทียบเท่า main.py
// ไม่ต้อง npm install: ใช้เฉพาะของที่มากับ Node.js 22.5 ขึ้นไป (node:http + node:sqlite)
//
// รัน:  API_KEY=รหัสลับ LLM_URL=http://127.0.0.1:9000/v1/chat/completions \
//       node --no-warnings server.mjs
// วาง persona.txt ไว้โฟลเดอร์เดียวกัน (บุคลิก Vex GPT จะถูกต่อเข้า /chat อัตโนมัติ)
// ใช้ไฟล์ memory.db เดิมจาก main.py ได้เลย (โครงสร้างเหมือนกัน)

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const env = process.env;

const PORT = Number(env.PORT || 8000);
const HOST = env.HOST || "0.0.0.0";
const DB_PATH = env.DB_PATH || path.join(__dirname, "memory.db");
const API_KEY = env.API_KEY || "change-me";
const LLM_URL = env.LLM_URL || ""; // OpenAI-compatible เช่น http://127.0.0.1:9000/v1/chat/completions
const LLM_KEY = env.LLM_KEY || "";
const LLM_MODEL = env.LLM_MODEL || "my-model";
// ค่าเริ่มต้นไม่เก็บคำตอบของ AI เองเป็นความจำ (กันความผิดสะสม) ตั้ง AUTO_SAVE=1 ถ้าต้องการ
const AUTO_SAVE = env.AUTO_SAVE === "1";
const CORS_ORIGIN = env.CORS_ORIGIN || "*";
const HISTORY_TURNS = 6;

// ── บุคลิก Vex GPT
let PERSONA = "";
try {
  PERSONA = fs.readFileSync(env.PERSONA_PATH || path.join(__dirname, "persona.txt"), "utf-8").trim();
} catch {}

// ── ฐานข้อมูล
const db = new DatabaseSync(DB_PATH);
const FTS_SQL = `CREATE VIRTUAL TABLE samples_fts
  USING fts5(prompt, response, content='samples', content_rowid='id', tokenize='trigram');`;

db.exec(`
CREATE TABLE IF NOT EXISTS samples(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  prompt TEXT,
  response TEXT NOT NULL,
  tag TEXT DEFAULT 'general',
  source TEXT DEFAULT 'manual',
  created REAL
);
${FTS_SQL.replace("CREATE VIRTUAL TABLE", "CREATE VIRTUAL TABLE IF NOT EXISTS")}
CREATE TRIGGER IF NOT EXISTS s_ai AFTER INSERT ON samples BEGIN
  INSERT INTO samples_fts(rowid, prompt, response) VALUES (new.id, new.prompt, new.response);
END;
CREATE TRIGGER IF NOT EXISTS s_ad AFTER DELETE ON samples BEGIN
  INSERT INTO samples_fts(samples_fts, rowid, prompt, response)
  VALUES ('delete', old.id, old.prompt, old.response);
END;
`);
// ฐานข้อมูลเก่าที่ไม่ได้ใช้ trigram → สร้างดัชนีใหม่ (ข้อมูลไม่หาย)
const ftsRow = db.prepare("SELECT sql FROM sqlite_master WHERE name='samples_fts'").get();
if (ftsRow && !(ftsRow.sql || "").includes("trigram")) {
  db.exec(`DROP TABLE samples_fts; ${FTS_SQL} INSERT INTO samples_fts(samples_fts) VALUES('rebuild');`);
}

if (API_KEY === "change-me") {
  console.log("⚠️  API_KEY ยังเป็นค่าเริ่มต้น 'change-me' — ตั้งรหัสด้วย API_KEY=... ก่อนเปิดให้คนอื่นเข้าถึง");
}

// ── ฟังก์ชันหลัก
const sha256 = (s) => crypto.createHash("sha256").update(s).digest();

function addSample({ prompt = "", response, tag = "general", source = "manual" }) {
  prompt = prompt ?? "";
  const hash = sha256(prompt + "\0" + response).toString("hex");
  const r = db
    .prepare("INSERT OR IGNORE INTO samples(hash,prompt,response,tag,source,created) VALUES(?,?,?,?,?,?)")
    .run(hash, prompt, response, tag, source, Date.now() / 1000);
  return r.changes ? { id: Number(r.lastInsertRowid), duplicate: false } : { duplicate: true };
}

// แตกข้อความเป็นชิ้นละ 3 ตัวอักษร (trigram) ใช้ค้นภาษาไทยที่ไม่มีเว้นวรรคได้
function ftsQuery(q, maxTerms = 60) {
  const seen = new Set();
  const terms = [];
  for (const chunk of q.replaceAll('"', " ").split(/\s+/)) {
    const cs = Array.from(chunk);
    for (let i = 0; i + 3 <= cs.length; i++) {
      const t = cs.slice(i, i + 3).join("");
      if (!seen.has(t)) {
        seen.add(t);
        terms.push(t);
      }
    }
  }
  return terms.slice(0, maxTerms).map((t) => `"${t}"`).join(" OR ");
}

function recall(q, k = 5) {
  const words = ftsQuery(q);
  let rows = [];
  if (words) {
    try {
      rows = db
        .prepare(
          "SELECT s.* FROM samples_fts f JOIN samples s ON s.id=f.rowid WHERE samples_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        .all(words, k);
    } catch {
      rows = [];
    }
  }
  if (!rows.length) {
    // ข้อความสั้นกว่า 3 ตัวอักษร → fallback LIKE
    const like = `%${q}%`;
    rows = db.prepare("SELECT * FROM samples WHERE prompt LIKE ? OR response LIKE ? LIMIT ?").all(like, like, k);
  }
  return rows.map((r) => ({ ...r }));
}

// ── แชท (จำอัตโนมัติ)
const HISTORY = new Map();

async function callLLM(messages) {
  if (!LLM_URL) return "(ยังไม่ได้ตั้ง LLM_URL) ความจำที่ดึงมา:\n" + messages[0].content;
  const r = await fetch(LLM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${LLM_KEY}` },
    body: JSON.stringify({ model: LLM_MODEL, messages }),
    signal: AbortSignal.timeout(120_000),
  });
  if (!r.ok) throw new Error(`LLM ตอบ ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const j = await r.json();
  return j.choices[0].message.content;
}

async function chat({ message, session = "default" }) {
  const mem = recall(message, 5);
  const memText = mem
    .map((m) => (m.prompt ? `- ${m.prompt} → ${m.response}` : `- ${m.response}`))
    .join("\n");
  const system =
    (PERSONA ? PERSONA + "\n\n" : "") +
    "ความรู้ที่เคยถูกสอนไว้ (ใช้ตอบถ้าเกี่ยวข้อง ถ้าไม่เกี่ยวให้ไม่ต้องพูดถึง):\n" +
    (memText || "(ไม่มี)");

  if (!HISTORY.has(session)) HISTORY.set(session, []);
  const hist = HISTORY.get(session);
  const msgs = [{ role: "system", content: system }];
  for (const [u, a] of hist.slice(-HISTORY_TURNS)) {
    msgs.push({ role: "user", content: u }, { role: "assistant", content: a });
  }
  msgs.push({ role: "user", content: message });

  const reply = await callLLM(msgs);
  hist.push([message, reply]);
  if (hist.length > HISTORY_TURNS) hist.splice(0, hist.length - HISTORY_TURNS);

  if (AUTO_SAVE) addSample({ prompt: message, response: reply, tag: "chat", source: `chat:${session}` });
  // ผู้ใช้สอนตรง ๆ เช่น "จำไว้ว่า ..." → เก็บเป็นความรู้ถาวร
  const text = message.trim();
  for (const key of ["จำไว้ว่า", "จำไว้", "remember that"]) {
    if (text.startsWith(key)) {
      const fact = text.slice(key.length).replace(/^[ :,]+|[ :,]+$/g, "");
      if (fact) addSample({ prompt: "", response: fact, tag: "taught", source: "user-taught" });
      break;
    }
  }
  return { reply, used_memory: mem.length };
}

// ── HTTP
class HttpError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

function send(res, status, body, type = "application/json; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": type,
    "Access-Control-Allow-Origin": CORS_ORIGIN,
    "Access-Control-Allow-Headers": "content-type, x-api-key",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
  });
  res.end(typeof body === "string" ? body : JSON.stringify(body));
}

function readJson(req, limit = 10 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > limit) {
        reject(new HttpError(413, "body too large"));
        req.destroy();
      } else chunks.push(c);
    });
    req.on("end", () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf-8")) : {});
      } catch {
        reject(new HttpError(400, "invalid json"));
      }
    });
    req.on("error", reject);
  });
}

const authed = (req) => {
  const got = sha256(String(req.headers["x-api-key"] ?? ""));
  return crypto.timingSafeEqual(got, sha256(API_KEY));
};

const int = (v, d) => (Number.isFinite(Number(v)) && v !== null && v !== "" ? Math.trunc(Number(v)) : d);

function validSample(s) {
  if (!s || typeof s.response !== "string" || !s.response) throw new HttpError(422, "response (string) required");
  return {
    prompt: typeof s.prompt === "string" ? s.prompt : "",
    response: s.response,
    tag: typeof s.tag === "string" ? s.tag : "general",
    source: typeof s.source === "string" ? s.source : "manual",
  };
}

function selectRows(tag, cols) {
  return tag
    ? db.prepare(`SELECT ${cols} FROM samples WHERE tag=?`).all(tag)
    : db.prepare(`SELECT ${cols} FROM samples`).all();
}

async function route(req, res) {
  const url = new URL(req.url, "http://x");
  const p = url.pathname;
  const q = url.searchParams;
  const m = req.method;

  if (m === "OPTIONS") return send(res, 204, "");
  if (m === "GET" && p === "/health") return send(res, 200, { ok: true });
  if (!authed(req)) throw new HttpError(401, "bad api key");

  if (m === "POST" && p === "/samples") return send(res, 200, addSample(validSample(await readJson(req))));

  if (m === "POST" && p === "/samples/bulk") {
    const items = await readJson(req);
    if (!Array.isArray(items)) throw new HttpError(422, "array required");
    const valid = items.map(validSample);
    const added = valid.reduce((n, s) => n + (addSample(s).duplicate ? 0 : 1), 0);
    return send(res, 200, { added, skipped: valid.length - added });
  }

  if (m === "GET" && p === "/samples") {
    const tag = q.get("tag");
    const limit = int(q.get("limit"), 50);
    const offset = int(q.get("offset"), 0);
    const rows = tag
      ? db.prepare("SELECT * FROM samples WHERE tag=? ORDER BY id DESC LIMIT ? OFFSET ?").all(tag, limit, offset)
      : db.prepare("SELECT * FROM samples ORDER BY id DESC LIMIT ? OFFSET ?").all(limit, offset);
    return send(res, 200, rows.map((r) => ({ ...r })));
  }

  let mm;
  if (m === "DELETE" && (mm = p.match(/^\/samples\/(\d+)$/))) {
    db.prepare("DELETE FROM samples WHERE id=?").run(Number(mm[1]));
    return send(res, 200, { deleted: Number(mm[1]) });
  }

  if (m === "GET" && p === "/recall") return send(res, 200, recall(q.get("q") ?? "", int(q.get("k"), 5)));

  if (m === "GET" && p === "/stats") {
    const total = db.prepare("SELECT COUNT(*) AS c FROM samples").get().c;
    const tags = {};
    for (const r of db.prepare("SELECT tag, COUNT(*) AS c FROM samples GROUP BY tag").all()) tags[r.tag] = r.c;
    return send(res, 200, { total: Number(total), tags });
  }

  if (m === "GET" && p === "/export/jsonl") {
    const asMessages = ["true", "1"].includes((q.get("messages") ?? "").toLowerCase());
    const lines = [];
    for (const r of selectRows(q.get("tag"), "prompt,response")) {
      if (asMessages) {
        if (!(r.prompt || "").trim()) continue;
        lines.push(JSON.stringify({ messages: [{ role: "user", content: r.prompt }, { role: "assistant", content: r.response }] }));
      } else lines.push(JSON.stringify({ prompt: r.prompt, response: r.response }));
    }
    return send(res, 200, lines.map((l) => l + "\n").join(""), "application/x-ndjson; charset=utf-8");
  }

  if (m === "GET" && p === "/export/nanogpt") {
    const parts = selectRows(q.get("tag"), "prompt,response").map((r) => {
      const pr = (r.prompt || "").trim();
      return (pr ? `User: ${pr}\nAI: ` : "") + r.response.trim();
    });
    return send(res, 200, parts.join("\n\n"), "text/plain; charset=utf-8");
  }

  if (m === "POST" && p === "/chat") {
    const b = await readJson(req);
    if (typeof b.message !== "string" || !b.message.trim()) throw new HttpError(422, "message (string) required");
    try {
      return send(res, 200, await chat({ message: b.message, session: typeof b.session === "string" ? b.session : "default" }));
    } catch (e) {
      if (e instanceof HttpError) throw e;
      throw new HttpError(502, String(e.message || e));
    }
  }

  throw new HttpError(404, "not found");
}

http
  .createServer((req, res) => {
    route(req, res).catch((e) => {
      if (e instanceof HttpError) return send(res, e.status, { detail: e.message });
      console.error(e);
      send(res, 500, { detail: "internal error" });
    });
  })
  .listen(PORT, HOST, () => console.log(`Memory server (Node ${process.version}) → http://${HOST}:${PORT}`));
