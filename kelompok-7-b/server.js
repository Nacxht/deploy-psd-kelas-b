const http = require("http");
const fs = require("fs");
const path = require("path");

const DEFAULT_PORT = 3000;
const WEB_DIR = path.join(__dirname, "web");

const MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon"
};

function serveFile(req, res) {
    let reqUrl = req.url.split("?")[0];
    if (reqUrl === "/") {
        reqUrl = "/index.html";
    }

    const safePath = path.normalize(reqUrl).replace(/^(\.\.[\/\\])+/, "");
    const filePath = path.join(WEB_DIR, safePath);

    if (!filePath.startsWith(WEB_DIR)) {
        res.writeHead(403, { "Content-Type": "text/plain" });
        res.end("403 Forbidden");
        return;
    }

    fs.stat(filePath, (err, stats) => {
        if (err || !stats.isFile()) {
            res.writeHead(404, { "Content-Type": "text/plain" });
            res.end("404 Not Found");
            return;
        }

        const ext = path.extname(filePath).toLowerCase();
        const contentType = MIME_TYPES[ext] || "application/octet-stream";

        res.writeHead(200, {
            "Content-Type": contentType,
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache"
        });

        const stream = fs.createReadStream(filePath);
        stream.pipe(res);
    });
}

function startServer(port) {
    const server = http.createServer(serveFile);

    server.on("error", (err) => {
        if (err.code === "EADDRINUSE") {
            console.log(`Port ${port} sedang digunakan, mencoba port ${port + 1}...`);
            startServer(port + 1);
        } else {
            console.error("Server error:", err);
        }
    });

    server.listen(port, () => {
        console.log("\n=======================================================");
        console.log("  🚀 SISTEM DETEKSI SUARA KEBOCORAN (LEAKSOUND)");
        console.log("=======================================================");
        console.log(`  Aplikasi berjalan di: http://localhost:${port}`);
        console.log("  Tekan Ctrl+C untuk menghentikan server.");
        console.log("=======================================================\n");
    });
}

startServer(DEFAULT_PORT);

