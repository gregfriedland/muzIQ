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
    this.demoButton = document.getElementById("demoButton");
    this.stopButton = document.getElementById("stopButton");
    this.resetButton = document.getElementById("resetButton");
    this.loadSpotify = document.getElementById("loadSpotify");
    this.spotifyUrl = document.getElementById("spotifyUrl");
    this.spotifyFrame = document.getElementById("spotifyFrame");
    this.instrumentRows = [];
    this.ws = null;
    this.captureContext = null;
    this.captureStream = null;
    this.captureNode = null;
    this.pendingFeatures = [];
    this.sendingFile = false;
    this.waitingForManualDemoPlay = false;
    this.demoAudio = null;
    this.demoInferenceStarted = false;
    this.noteActive = false;
    this.activeRowKey = null;
    this.lastSignalAt = 0;
    this.lastPredictionAt = 0;
    this.bind();
    this.resize();
    window.addEventListener("resize", () => this.resize());
    window.setInterval(() => this.clearRowsAfterSilence(performance.now()), 250);
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
    this.filePlayer.addEventListener("play", () => this.handleAudioPlay());
    this.captureButton.addEventListener("click", () => this.startCapture());
    this.demoButton.addEventListener("click", () => this.playDemoBeat());
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
      if (!this.waitingForManualDemoPlay) {
        this.setStatus("Inference ready", "ready");
      }
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
    const feature = this.pendingFeatures.shift() || emptyAudioFeature();
    this.updateInstrumentRows(payload, feature);
    this.sourceCount.textContent = String(this.instrumentRows.length);
    this.frameCount.textContent = String(payload.frame_count);
    this.latency.textContent = `${Math.max(0, performance.now() - this.lastSentAt).toFixed(0)} ms`;
    this.renderCards();
  }

  updateInstrumentRows(payload, feature) {
    const now = performance.now();
    if (feature.rms > 0.006) {
      this.lastSignalAt = now;
    }
    if (this.noteActive && feature.rms <= 0.012) {
      this.hideActiveNote();
    }
    if (!this.noteActive && feature.rms >= 0.026) {
      const source = chooseSourceForFeature(payload.sources, feature);
      if (source) {
        this.showNote(source, feature, now);
      }
    }
    this.clearRowsAfterSilence(now);
  }

  showNote(source, feature, now) {
    const key = `slot:${source.slot}`;
    let row = this.instrumentRows.find((candidate) => candidate.key === key);
    if (!row) {
      row = {
        key,
        slot: source.slot,
        family: source.family,
        color: ROW_COLORS[source.slot % ROW_COLORS.length],
        minFrequency: feature.meanFrequency,
        maxFrequency: feature.meanFrequency,
        lastSeenAt: now,
        activeNote: null,
      };
      this.instrumentRows.push(row);
    }
    row.family = source.family;
    row.lastSeenAt = now;
    row.minFrequency = Math.min(row.minFrequency, feature.meanFrequency);
    row.maxFrequency = Math.max(row.maxFrequency, feature.meanFrequency);
    row.activeNote = {
      frequency: feature.meanFrequency,
      activity: Math.max(source.activity, feature.rms * 12),
    };
    this.noteActive = true;
    this.activeRowKey = key;
  }

  hideActiveNote() {
    const row = this.instrumentRows.find((candidate) => candidate.key === this.activeRowKey);
    if (row) {
      row.activeNote = null;
    }
    this.noteActive = false;
    this.activeRowKey = null;
  }

  clearRowsAfterSilence(now) {
    if (!this.instrumentRows.length || !this.lastSignalAt) {
      return;
    }
    if (now - this.lastSignalAt < 3000) {
      return;
    }
    this.instrumentRows = [];
    this.noteActive = false;
    this.activeRowKey = null;
    this.sourceCount.textContent = "0";
    this.renderCards();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send("reset");
    }
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

  async playDemoBeat() {
    this.stopAudio();
    await this.connect();
    this.reset();
    const audio = makeAlternatingBeat(TARGET_SAMPLE_RATE, 8);
    this.demoAudio = audio;
    this.demoInferenceStarted = false;
    const isPlaying = await this.playDemoAudio(audio);
    this.waitingForManualDemoPlay = !isPlaying;
    this.setStatus(
      isPlaying ? "Playing demo beat" : "Press play in the audio control",
      isPlaying ? "ready" : "error",
    );
    if (isPlaying && !this.demoInferenceStarted) {
      this.startDemoInference();
    }
  }

  handleAudioPlay() {
    if (!this.demoAudio || this.demoInferenceStarted) {
      return;
    }
    this.waitingForManualDemoPlay = false;
    this.setStatus("Playing demo beat", "ready");
    this.startDemoInference();
  }

  startDemoInference() {
    if (!this.demoAudio || this.demoInferenceStarted) {
      return;
    }
    this.demoInferenceStarted = true;
    this.sendingFile = true;
    void this.streamAudioArray(this.demoAudio);
  }

  async playDemoAudio(audio) {
    const wav = audioToWavBlob(audio, TARGET_SAMPLE_RATE);
    if (this.filePlayer.src) {
      URL.revokeObjectURL(this.filePlayer.src);
    }
    this.filePlayer.src = URL.createObjectURL(wav);
    this.filePlayer.currentTime = 0;
    this.filePlayer.volume = 1;
    this.filePlayer.muted = false;
    try {
      await this.filePlayer.play();
      return !this.filePlayer.paused;
    } catch (_error) {
      return false;
    }
  }

  sendChunk(chunk) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !chunk.length) {
      return;
    }
    this.pendingFeatures.push(analyzeAudioChunk(chunk, TARGET_SAMPLE_RATE));
    if (this.pendingFeatures.length > 12) {
      this.pendingFeatures.shift();
    }
    this.lastSentAt = performance.now();
    this.ws.send(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
  }

  stopAudio() {
    this.sendingFile = false;
    this.waitingForManualDemoPlay = false;
    this.demoAudio = null;
    this.demoInferenceStarted = false;
    this.pendingFeatures = [];
    this.hideActiveNote();
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
    this.instrumentRows = [];
    this.pendingFeatures = [];
    this.noteActive = false;
    this.activeRowKey = null;
    this.lastSignalAt = 0;
    this.sourceCount.textContent = "0";
    this.frameCount.textContent = "0";
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
    this.sourceRows.replaceChildren(...this.sortedRows().map((source) => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.style.borderColor = colorCss(source.color, source.activeNote ? 1 : 0.25);
      const title = document.createElement("strong");
      title.textContent = `ID ${source.slot + 1}`;
      const detail = document.createElement("span");
      detail.textContent = `${source.family}  ${frequencyRangeLabel(source)}`;
      card.append(title, detail);
      return card;
    }));
  }

  sortedRows() {
    return [...this.instrumentRows].sort((left, right) => right.lastSeenAt - left.lastSeenAt);
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
    const rows = this.sortedRows();
    const rowCount = Math.max(1, rows.length);
    const rowHeight = Math.floor(height / rowCount);
    rows.forEach((row, index) => this.drawRow(row, index * rowHeight, rowHeight, width));
    requestAnimationFrame(() => this.draw());
  }

  drawRow(row, y, rowHeight, width) {
    const color = row.color;
    this.ctx.fillStyle = "rgba(255,255,255,0.05)";
    this.ctx.fillRect(0, y + rowHeight - 1, width, 1);
    this.ctx.fillStyle = "rgba(237,242,247,0.72)";
    this.ctx.font = `${Math.max(11, Math.floor(rowHeight * 0.12))}px sans-serif`;
    this.ctx.fillText(`ID ${row.slot + 1} ${row.family}`, 12, y + 24);
    if (row.activeNote) {
      const boxSize = Math.max(18, Math.min(54, Math.floor(rowHeight * 0.48)));
      const labelWidth = Math.min(170, Math.floor(width * 0.18));
      const usableWidth = Math.max(1, width - labelWidth - boxSize - 20);
      const x = labelWidth + frequencyPosition(row, row.activeNote.frequency) * usableWidth;
      const centerY = y + rowHeight / 2;
      this.ctx.fillStyle = colorCss(color, Math.min(1, 0.35 + row.activeNote.activity));
      this.ctx.fillRect(x, centerY - boxSize / 2, boxSize, boxSize);
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

function analyzeAudioChunk(input, sampleRate) {
  let energy = 0;
  for (let i = 0; i < input.length; i += 1) {
    energy += input[i] * input[i];
  }
  const rms = Math.sqrt(energy / Math.max(1, input.length));
  if (rms < 0.000001) {
    return { rms, meanFrequency: 0 };
  }
  const windowSize = Math.min(512, input.length);
  let weightedFrequency = 0;
  let weight = 0;
  for (let bin = 1; bin < windowSize / 2; bin += 1) {
    let real = 0;
    let imag = 0;
    for (let index = 0; index < windowSize; index += 1) {
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, windowSize - 1));
      const phase = (2 * Math.PI * bin * index) / windowSize;
      const sample = input[index] * window;
      real += sample * Math.cos(phase);
      imag -= sample * Math.sin(phase);
    }
    const magnitude = Math.sqrt(real * real + imag * imag);
    const frequency = (bin * sampleRate) / windowSize;
    weightedFrequency += frequency * magnitude;
    weight += magnitude;
  }
  return {
    rms,
    meanFrequency: Math.max(20, Math.min(sampleRate / 2, weightedFrequency / Math.max(weight, 0.000001))),
  };
}

function emptyAudioFeature() {
  return { rms: 0, meanFrequency: 0 };
}

function chooseSourceForFeature(sources, feature) {
  const active = sources
    .filter((source) => source.activity >= 0.35)
    .sort((left, right) => right.confidence - left.confidence);
  if (!active.length) {
    return sources.slice().sort((left, right) => right.activity - left.activity)[0] || null;
  }
  if (feature.meanFrequency > 0 && feature.meanFrequency < 280) {
    return active.find((source) => source.family === "bass") || active[0];
  }
  if (feature.meanFrequency >= 280) {
    return active.find((source) => source.family !== "bass") || active[0];
  }
  return active[0];
}

function frequencyPosition(row, frequency) {
  const range = row.maxFrequency - row.minFrequency;
  if (range < 20) {
    return 0.5;
  }
  return Math.max(0, Math.min(1, (frequency - row.minFrequency) / range));
}

function frequencyRangeLabel(row) {
  if (row.maxFrequency <= 0) {
    return "-";
  }
  if (row.maxFrequency - row.minFrequency < 20) {
    return `${Math.round(row.maxFrequency)} Hz`;
  }
  return `${Math.round(row.minFrequency)}-${Math.round(row.maxFrequency)} Hz`;
}

function makeAlternatingBeat(sampleRate, seconds) {
  const length = Math.floor(sampleRate * seconds);
  const output = new Float32Array(length);
  for (let beat = 0; beat < seconds * 2; beat += 1) {
    const start = Math.floor(beat * 0.5 * sampleRate);
    if (beat % 2 === 0) {
      addKick(output, sampleRate, start);
    } else {
      addPing(output, sampleRate, start);
    }
  }
  return output;
}

function audioToWavBlob(audio, sampleRate) {
  const bytesPerSample = 2;
  const headerBytes = 44;
  const buffer = new ArrayBuffer(headerBytes + audio.length * bytesPerSample);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + audio.length * bytesPerSample, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, audio.length * bytesPerSample, true);
  for (let i = 0; i < audio.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, audio[i]));
    view.setInt16(headerBytes + i * bytesPerSample, sample * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function addKick(output, sampleRate, start) {
  const length = Math.floor(sampleRate * 0.22);
  for (let i = 0; i < length && start + i < output.length; i += 1) {
    const t = i / sampleRate;
    const envelope = Math.exp(-t * 16);
    const frequency = 95 - 38 * Math.min(1, t / 0.18);
    output[start + i] += 0.72 * envelope * Math.sin(2 * Math.PI * frequency * t);
  }
}

function addPing(output, sampleRate, start) {
  const length = Math.floor(sampleRate * 0.16);
  for (let i = 0; i < length && start + i < output.length; i += 1) {
    const t = i / sampleRate;
    const envelope = Math.exp(-t * 24);
    const tone = Math.sin(2 * Math.PI * 760 * t) + 0.35 * Math.sin(2 * Math.PI * 1520 * t);
    output[start + i] += 0.42 * envelope * tone;
  }
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
