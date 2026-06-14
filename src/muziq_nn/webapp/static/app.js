const TARGET_SAMPLE_RATE = 16000;
const ROW_COLORS = [
  [180, 60, 255],
  [40, 120, 255],
  [0, 220, 130],
  [255, 180, 0],
  [255, 60, 60],
];

class SourceGridApp {
  constructor() {
    this.canvas = document.getElementById("sourceGrid");
    this.ctx = this.canvas.getContext("2d");
    this.status = document.getElementById("status");
    this.statusDot = document.getElementById("statusDot");
    this.checkpointPath = document.getElementById("checkpointPath");
    this.sourceRows = document.getElementById("sourceRows");
    this.sourceCount = document.getElementById("sourceCount");
    this.frameCount = document.getElementById("frameCount");
    this.latency = document.getElementById("latency");
    this.fileInput = document.getElementById("fileInput");
    this.filePlayer = document.getElementById("filePlayer");
    this.captureButton = document.getElementById("captureButton");
    this.stopButton = document.getElementById("stopButton");
    this.resetButton = document.getElementById("resetButton");
    this.loadSpotify = document.getElementById("loadSpotify");
    this.spotifyUrl = document.getElementById("spotifyUrl");
    this.spotifyFrame = document.getElementById("spotifyFrame");
    this.history = Array.from({ length: 5 }, () => []);
    this.sources = Array.from({ length: 5 }, (_, slot) => ({
      slot,
      activity: 0,
      family: "-",
      confidence: 0,
      position: 0.5,
    }));
    this.ws = null;
    this.captureContext = null;
    this.captureStream = null;
    this.captureNode = null;
    this.sendingFile = false;
    this.lastPredictionAt = 0;
    this.bind();
    this.resize();
    window.addEventListener("resize", () => this.resize());
    requestAnimationFrame(() => this.draw());
  }

  async start() {
    const status = await fetch("/api/status").then((res) => res.json());
    if (status.checkpoint_path) {
      this.checkpointPath.value = status.checkpoint_path;
      this.setStatus("Checkpoint ready", "ready");
    } else {
      this.setStatus("Checkpoint missing", "error");
    }
    this.renderCards();
  }

  bind() {
    this.fileInput.addEventListener("change", () => this.handleFile());
    this.captureButton.addEventListener("click", () => this.startCapture());
    this.stopButton.addEventListener("click", () => this.stopAudio());
    this.resetButton.addEventListener("click", () => this.reset());
    this.loadSpotify.addEventListener("click", () => this.loadSpotifyEmbed());
  }

  async connect() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return this.ws;
    }
    const params = new URLSearchParams();
    if (this.checkpointPath.value.trim()) {
      params.set("checkpoint", this.checkpointPath.value.trim());
    }
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${scheme}://${window.location.host}/ws/infer?${params}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.addEventListener("message", (event) => this.handleServerMessage(event));
    await new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error("WebSocket timeout")), 5000);
      this.ws.addEventListener("open", () => {
        window.clearTimeout(timeout);
        resolve();
      }, { once: true });
      this.ws.addEventListener("error", reject, { once: true });
    });
    return this.ws;
  }

  handleServerMessage(event) {
    const payload = JSON.parse(event.data);
    if (payload.type === "ready") {
      this.setStatus("Inference ready", "ready");
      return;
    }
    if (payload.type === "error") {
      this.setStatus(payload.message, "error");
      return;
    }
    if (payload.type !== "prediction") {
      return;
    }
    this.lastPredictionAt = performance.now();
    this.sourceCount.textContent = String(payload.source_count);
    this.frameCount.textContent = String(payload.frame_count);
    this.latency.textContent = `${Math.max(0, performance.now() - this.lastSentAt).toFixed(0)} ms`;
    for (const source of payload.sources) {
      this.sources[source.slot] = source;
      this.history[source.slot].push(source.activity);
      if (this.history[source.slot].length > this.canvas.width) {
        this.history[source.slot].shift();
      }
    }
    this.renderCards();
  }

  async handleFile() {
    const file = this.fileInput.files?.[0];
    if (!file) {
      return;
    }
    this.stopAudio();
    this.filePlayer.src = URL.createObjectURL(file);
    const audio = await this.decodeFile(file);
    await this.connect();
    this.sendingFile = true;
    this.filePlayer.play().catch(() => {});
    this.streamAudioArray(audio);
  }

  async decodeFile(file) {
    const buffer = await file.arrayBuffer();
    const context = new AudioContext();
    const decoded = await context.decodeAudioData(buffer);
    const mono = this.mixToMono(decoded);
    await context.close();
    return resampleLinear(mono, decoded.sampleRate, TARGET_SAMPLE_RATE);
  }

  mixToMono(decoded) {
    const samples = decoded.length;
    const mono = new Float32Array(samples);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const source = decoded.getChannelData(channel);
      for (let index = 0; index < samples; index += 1) {
        mono[index] += source[index] / decoded.numberOfChannels;
      }
    }
    return mono;
  }

  async streamAudioArray(audio) {
    const chunkSize = 1600;
    let offset = 0;
    while (this.sendingFile && offset < audio.length) {
      const chunk = audio.slice(offset, Math.min(audio.length, offset + chunkSize));
      this.sendChunk(chunk);
      offset += chunkSize;
      await sleep((chunk.length / TARGET_SAMPLE_RATE) * 1000);
    }
    this.sendingFile = false;
  }

  async startCapture() {
    this.stopAudio();
    await this.connect();
    this.captureStream = await navigator.mediaDevices.getDisplayMedia({
      audio: true,
      video: true,
    });
    const audioTracks = this.captureStream.getAudioTracks();
    if (!audioTracks.length) {
      this.setStatus("No captured audio track", "error");
      return;
    }
    this.captureContext = new AudioContext();
    const source = this.captureContext.createMediaStreamSource(this.captureStream);
    this.captureNode = this.captureContext.createScriptProcessor(4096, 1, 1);
    this.captureNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      this.sendChunk(resampleLinear(input, this.captureContext.sampleRate, TARGET_SAMPLE_RATE));
    };
    const sink = this.captureContext.createGain();
    sink.gain.value = 0;
    source.connect(this.captureNode);
    this.captureNode.connect(sink);
    sink.connect(this.captureContext.destination);
    this.setStatus("Capturing audio", "ready");
  }

  sendChunk(chunk) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !chunk.length) {
      return;
    }
    this.lastSentAt = performance.now();
    this.ws.send(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
  }

  stopAudio() {
    this.sendingFile = false;
    if (this.captureNode) {
      this.captureNode.disconnect();
      this.captureNode = null;
    }
    if (this.captureContext) {
      this.captureContext.close();
      this.captureContext = null;
    }
    if (this.captureStream) {
      this.captureStream.getTracks().forEach((track) => track.stop());
      this.captureStream = null;
    }
    if (this.filePlayer.src) {
      this.filePlayer.pause();
    }
  }

  reset() {
    this.history = Array.from({ length: 5 }, () => []);
    this.sources = Array.from({ length: 5 }, (_, slot) => ({
      slot,
      activity: 0,
      family: "-",
      confidence: 0,
      position: 0.5,
    }));
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send("reset");
    }
    this.renderCards();
  }

  loadSpotifyEmbed() {
    const trackId = spotifyTrackId(this.spotifyUrl.value.trim());
    if (!trackId) {
      this.setStatus("Invalid Spotify track URL", "error");
      return;
    }
    const iframe = document.createElement("iframe");
    iframe.allow = "autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture";
    iframe.loading = "lazy";
    iframe.src = `https://open.spotify.com/embed/track/${trackId}`;
    this.spotifyFrame.replaceChildren(iframe);
    this.setStatus("Spotify track loaded", "ready");
  }

  renderCards() {
    this.sourceRows.replaceChildren(...this.sources.map((source, slot) => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.style.borderColor = colorCss(ROW_COLORS[slot], Math.max(0.25, source.activity));
      const title = document.createElement("strong");
      title.textContent = `ID ${slot + 1}`;
      const detail = document.createElement("span");
      detail.textContent = `${source.family}  ${(source.activity * 100).toFixed(0)}%`;
      card.append(title, detail);
      return card;
    }));
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(320, Math.floor(rect.width * ratio));
    this.canvas.height = Math.max(220, Math.floor(rect.height * ratio));
  }

  draw() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    this.ctx.fillStyle = "#050607";
    this.ctx.fillRect(0, 0, width, height);
    const rowHeight = Math.floor(height / 5);
    for (let slot = 0; slot < 5; slot += 1) {
      const y = slot * rowHeight;
      this.drawRow(slot, y, rowHeight, width);
    }
    requestAnimationFrame(() => this.draw());
  }

  drawRow(slot, y, rowHeight, width) {
    const history = this.history[slot];
    const color = ROW_COLORS[slot];
    this.ctx.fillStyle = "rgba(255,255,255,0.05)";
    this.ctx.fillRect(0, y + rowHeight - 1, width, 1);
    for (let i = 0; i < history.length; i += 1) {
      const level = Math.min(1, Math.max(0, history[i]));
      const x = width - history.length + i;
      this.ctx.fillStyle = colorCss(color, level);
      this.ctx.fillRect(x, y + 4, 1, Math.max(1, rowHeight - 8));
    }
    const source = this.sources[slot];
    const boxWidth = Math.max(12, Math.floor(width * 0.035));
    const x = Math.max(0, Math.min(width - boxWidth, Math.floor(source.position * width)));
    const pulse = Math.max(source.activity, source.onset || 0);
    if (pulse > 0.05) {
      this.ctx.fillStyle = colorCss(color, Math.min(1, 0.25 + pulse));
      this.ctx.fillRect(x, y + 6, boxWidth, Math.max(1, rowHeight - 12));
    }
  }

  setStatus(message, state) {
    this.status.textContent = message;
    this.statusDot.className = `status-dot ${state || ""}`;
  }
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return new Float32Array(input);
  }
  const ratio = inputRate / outputRate;
  const outputLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(input.length - 1, left + 1);
    const frac = position - left;
    output[i] = input[left] * (1 - frac) + input[right] * frac;
  }
  return output;
}

function spotifyTrackId(value) {
  const uri = value.match(/^spotify:track:([A-Za-z0-9]+)$/);
  if (uri) {
    return uri[1];
  }
  try {
    const url = new URL(value);
    const parts = url.pathname.split("/").filter(Boolean);
    const trackIndex = parts.indexOf("track");
    if (trackIndex >= 0 && parts[trackIndex + 1]) {
      return parts[trackIndex + 1];
    }
  } catch (_error) {
    return null;
  }
  return null;
}

function colorCss(rgb, level) {
  const alpha = Math.max(0.08, Math.min(1, level));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

new SourceGridApp().start();
