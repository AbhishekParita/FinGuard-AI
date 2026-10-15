import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { access, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const WEB_ROOT = path.join(ROOT, "frontend");
const API_URL = process.env.FINGUARD_API_URL || "http://127.0.0.1:8000";
const WEB_HOST = process.env.FINGUARD_WEB_HOST || "127.0.0.1";
const WEB_PORT = Number(process.env.FINGUARD_WEB_PORT || 5173);
const AUTO_START_API = process.env.FINGUARD_API_AUTOSTART !== "false";
let apiProcess;

async function loadLocalEnv() {
  try {
    const content = await readFile(path.join(ROOT, ".env"), "utf8");
    for (const line of content.split(/\r?\n/)) {
      const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
      if (match && process.env[match[1]] === undefined) {
        process.env[match[1]] = match[2].replace(/^(?:"(.*)"|'(.*)')$/, (_, doubleQuoted, singleQuoted) => doubleQuoted ?? singleQuoted);
      }
    }
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

async function apiIsReady() {
  try {
    const response = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(800) });
    return response.ok;
  } catch {
    return false;
  }
}

async function startLocalApi() {
  if (await apiIsReady()) {
    console.log(`Using existing local API at ${API_URL}`);
    return;
  }
  if (!AUTO_START_API) {
    for (let attempt = 0; attempt < 40; attempt += 1) {
      if (await apiIsReady()) {
        console.log(`Connected to local API at ${API_URL}`);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error("Could not connect to the configured local API.");
  }
  const virtualEnvPython = path.join(ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  let python = process.platform === "win32" ? "python" : "python3";
  try {
    await access(virtualEnvPython, constants.X_OK);
    python = virtualEnvPython;
  } catch {
    // Fall back to Python on PATH when the project virtual environment is absent.
  }
  apiProcess = spawn(python, ["-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", new URL(API_URL).port || "8000"], {
    cwd: ROOT,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
  });
  apiProcess.on("error", (error) => console.error(`Could not start FastAPI: ${error.message}`));
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (await apiIsReady()) {
      console.log(`Started local API at ${API_URL}`);
      return;
    }
    if (apiProcess.exitCode !== null) break;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("FastAPI did not start. Check the Python environment and generated data.");
}

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${WEB_HOST}:${WEB_PORT}`);
    if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
      const target = `${API_URL}${url.pathname.slice(4)}${url.search}`;
      const body = ["GET", "HEAD"].includes(request.method || "GET") ? undefined : await new Promise((resolve, reject) => {
        const chunks = [];
        request.on("data", (chunk) => chunks.push(chunk));
        request.on("end", () => resolve(Buffer.concat(chunks)));
        request.on("error", reject);
      });
      const upstream = await fetch(target, {
        method: request.method,
        headers: { "content-type": request.headers["content-type"] || "application/json", ...(request.headers.cookie ? { cookie: request.headers.cookie } : {}) },
        body,
      });
      const headers = { "content-type": upstream.headers.get("content-type") || "application/json" };
      const setCookie = upstream.headers.get("set-cookie");
      if (setCookie) headers["set-cookie"] = setCookie;
      response.writeHead(upstream.status, headers);
      response.end(Buffer.from(await upstream.arrayBuffer()));
      return;
    }

    if (url.pathname === "/favicon.ico") {
      response.writeHead(204).end();
      return;
    }

    let relative = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
    let filePath = path.resolve(WEB_ROOT, `.${relative}`);
    if (!filePath.startsWith(`${WEB_ROOT}${path.sep}`) && filePath !== path.join(WEB_ROOT, "index.html")) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    try {
      const content = await readFile(filePath);
      response.writeHead(200, { "content-type": mimeTypes[path.extname(filePath)] || "application/octet-stream" });
      response.end(content);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const content = await readFile(path.join(WEB_ROOT, "index.html"));
      response.writeHead(200, { "content-type": mimeTypes[".html"] });
      response.end(content);
    }
  } catch (error) {
    response.writeHead(502, { "content-type": "application/json; charset=utf-8" });
    response.end(JSON.stringify({ detail: error.message || "Local service unavailable" }));
  }
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    server.close();
    if (apiProcess && apiProcess.exitCode === null) apiProcess.kill();
  });
}

await loadLocalEnv();
process.env.AGNO_TELEMETRY ??= "false";
await startLocalApi();
server.listen(WEB_PORT, WEB_HOST, () => {
  console.log(`FinGuard AI is ready at http://${WEB_HOST}:${WEB_PORT}`);
});
