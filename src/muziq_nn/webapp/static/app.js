const ROW_COLORS = [
  [180, 60, 255],
  [40, 120, 255],
  [0, 220, 130],
  [255, 180, 0],
  [255, 60, 60],
];
const BLACKHOLE_CAPTURE_DEVICE = "BlackHole 2ch";
const SIGNAL_ACTIVE_RMS = 0.006;
const SOURCE_ONSET_THRESHOLD = 0.35;
const SOURCE_OFFSET_THRESHOLD = 0.35;
const SOURCE_ONSET_REFRACTORY_MS = 200;
const SOURCE_OFFSET_REFRACTORY_MS = 200;

class SourceGridApp {
  constructor() {
    this.canvas = document.getElementById("sourceGrid");
    this.ctx = this.canvas.getContext("2d");
    this.pixelRatio = 1;
    this.canvasCssWidth = 960;
    this.canvasCssHeight = 360;
    this.status = document.getElementById("status");
    this.statusDot = document.getElementById("statusDot");
    this.checkpointPath = document.getElementById("checkpointPath");
    this.sourceRows = document.getElementById("sourceRows");
    this.sourceCount = document.getElementById("sourceCount");
    this.frameCount = document.getElementById("frameCount");
    this.latency = document.getElementById("latency");
    this.signalLevel = document.getElementById("signalLevel");
    this.blackholeButton = document.getElementById("blackholeButton");
    this.stopButton = document.getElementById("stopButton");
    this.resetButton = document.getElementById("resetButton");
    this.instrumentRows = [];
    this.ws = null;
    this.noteActive = false;
    this.activeRowKey = null;
    this.slotStates = new Map();
    this.lastSignalAt = 0;
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
    this.blackholeButton.addEventListener("click", () => {
      this.startServerCapture(BLACKHOLE_CAPTURE_DEVICE);
    });
    this.stopButton.addEventListener("click", () => this.stopAudio());
    this.resetButton.addEventListener("click", () => this.reset());
  }

  startServerCapture(captureDevice) {
    this.reset();
    this.closeWebSocket();
    const params = new URLSearchParams();
    if (this.checkpointPath.value.trim()) {
      params.set("checkpoint", this.checkpointPath.value.trim());
    }
    params.set("capture_device", captureDevice);
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${scheme}://${window.location.host}/ws/capture?${params}`);
    this.ws.addEventListener("message", (event) => this.handleServerMessage(event));
    this.ws.addEventListener("error", () => this.setStatus("BlackHole capture failed", "error"));
    this.ws.addEventListener("close", () => {
      if (this.status.textContent.startsWith("Capturing")) {
        this.setStatus("BlackHole capture stopped", "error");
      }
    });
    this.setStatus(`Capturing ${captureDevice}`, "ready");
  }

  handleServerMessage(event) {
    const payload = JSON.parse(event.data);
    if (payload.type === "ready") {
      this.setStatus(`Capturing ${payload.capture_device || BLACKHOLE_CAPTURE_DEVICE}`, "ready");
      return;
    }
    if (payload.type === "error") {
      this.setStatus(payload.message, "error");
      return;
    }
    if (payload.type !== "prediction") {
      return;
    }
    const feature = serverAudioFeature(payload);
    this.signalLevel.textContent = signalLevelLabel(feature.rms);
    this.updateInstrumentRows(payload, feature);
    this.sourceCount.textContent = String(this.instrumentRows.length);
    this.frameCount.textContent = String(payload.frame_count || 0);
    this.latency.textContent = latencyLabel(payload.latency_ms);
    this.renderCards();
  }

  updateInstrumentRows(payload, feature) {
    const now = performance.now();
    if (feature.rms > SIGNAL_ACTIVE_RMS) {
      this.lastSignalAt = now;
    }

    if (feature.rms <= SIGNAL_ACTIVE_RMS) {
      for (const row of this.instrumentRows) {
        this.stopRowNote(row);
      }
      this.noteActive = false;
      this.activeRowKey = null;
      this.slotStates.clear();
      this.clearRowsAfterSilence(now);
      return;
    }

    if (!payload.sources.length || payload.frame_count === 0) {
      this.clearRowsAfterSilence(now);
      return;
    }

    for (const source of payload.sources) {
      this.updateSlotState(source, now);
    }
    this.noteActive = this.instrumentRows.some((row) => row.activeNote);
    this.activeRowKey = this.instrumentRows.find((row) => row.activeNote)?.key || null;
    this.clearRowsAfterSilence(now);
  }

  updateSlotState(source, now) {
    const key = sourceKey(source);
    let state = this.slotStates.get(key);
    if (!state) {
      state = {
        previous: null,
        current: null,
        lastOnsetPeakAt: Number.NEGATIVE_INFINITY,
        lastOffsetPeakAt: Number.NEGATIVE_INFINITY,
      };
      this.slotStates.set(key, state);
    }
    const previous = state.previous;
    const current = state.current;
    if (current && isLocalPeak(previous, current, source, "onset", SOURCE_ONSET_THRESHOLD)) {
      if (now - state.lastOnsetPeakAt >= SOURCE_ONSET_REFRACTORY_MS) {
        this.startSourceNote(current, now);
        state.lastOnsetPeakAt = now;
      }
    }
    if (current && isLocalPeak(previous, current, source, "offset", SOURCE_OFFSET_THRESHOLD)) {
      if (now - state.lastOffsetPeakAt >= SOURCE_OFFSET_REFRACTORY_MS) {
        this.stopSourceNote(current);
        state.lastOffsetPeakAt = now;
      }
    }
    state.previous = current;
    state.current = source;
  }

  startSourceNote(source, now) {
    const key = sourceKey(source);
    let row = this.instrumentRows.find((candidate) => candidate.key === key);
    if (!row) {
      row = {
        key,
        slot: source.slot,
        color: ROW_COLORS[source.slot % ROW_COLORS.length],
        lastSeenAt: now,
        lastNoteStartAt: Number.NEGATIVE_INFINITY,
        activeNote: null,
      };
      this.instrumentRows.push(row);
    }
    this.showNote(row, source, now);
  }

  stopSourceNote(source) {
    const row = this.instrumentRows.find((candidate) => candidate.key === sourceKey(source));
    if (row) {
      this.stopRowNote(row);
    }
  }

  showNote(row, source, now) {
    row.lastSeenAt = now;
    row.activeNote = {
      position: clampUnit(source.position),
      activity: Math.max(source.onset, 0.35),
    };
    row.lastNoteStartAt = now;
    this.noteActive = true;
    this.activeRowKey = row.key;
  }

  stopRowNote(row) {
    row.activeNote = null;
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
    this.slotStates.clear();
    this.sourceCount.textContent = "0";
    this.renderCards();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send("reset");
    }
  }

  stopAudio() {
    for (const row of this.instrumentRows) {
      this.stopRowNote(row);
    }
    this.noteActive = false;
    this.activeRowKey = null;
    this.closeWebSocket();
    this.setStatus("BlackHole capture stopped", "ready");
    this.renderCards();
  }

  closeWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  reset() {
    this.instrumentRows = [];
    this.noteActive = false;
    this.activeRowKey = null;
    this.slotStates = new Map();
    this.lastSignalAt = 0;
    this.sourceCount.textContent = "0";
    this.frameCount.textContent = "0";
    this.latency.textContent = "-";
    this.signalLevel.textContent = "-";
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send("reset");
    }
    this.renderCards();
  }

  renderCards() {
    this.sourceRows.replaceChildren(...this.sortedRows().map((source) => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.style.borderColor = colorCss(source.color, source.activeNote ? 1 : 0.25);
      const title = document.createElement("strong");
      title.textContent = `ID ${source.slot + 1}`;
      card.append(title);
      return card;
    }));
  }

  sortedRows() {
    return [...this.instrumentRows].sort((left, right) => right.lastSeenAt - left.lastSeenAt);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.pixelRatio = window.devicePixelRatio || 1;
    this.canvasCssWidth = Math.max(320, Math.floor(rect.width));
    this.canvasCssHeight = Math.max(220, Math.floor(rect.height));
    this.canvas.width = Math.floor(this.canvasCssWidth * this.pixelRatio);
    this.canvas.height = Math.floor(this.canvasCssHeight * this.pixelRatio);
  }

  draw() {
    this.ctx.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
    const width = this.canvasCssWidth;
    const height = this.canvasCssHeight;
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
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(0, y, width, rowHeight);
    this.ctx.clip();
    this.ctx.fillStyle = "rgba(255,255,255,0.05)";
    this.ctx.fillRect(0, y + rowHeight - 1, width, 1);
    this.ctx.fillStyle = "rgba(237,242,247,0.72)";
    this.ctx.textBaseline = "top";
    this.ctx.font = `${Math.max(11, Math.min(16, Math.floor(rowHeight * 0.14)))}px sans-serif`;
    this.ctx.fillText(`ID ${row.slot + 1}`, 12, y + 10);
    if (row.activeNote) {
      const boxSize = Math.max(18, Math.min(54, Math.floor(rowHeight * 0.48)));
      const labelWidth = Math.min(170, Math.floor(width * 0.18));
      const usableWidth = Math.max(1, width - labelWidth - boxSize - 20);
      const x = labelWidth + row.activeNote.position * usableWidth;
      const centerY = y + rowHeight / 2;
      this.ctx.fillStyle = colorCss(color, Math.min(1, 0.35 + row.activeNote.activity));
      this.ctx.fillRect(x, centerY - boxSize / 2, boxSize, boxSize);
    }
    this.ctx.restore();
  }

  setStatus(message, state) {
    this.status.textContent = message;
    this.statusDot.className = `status-dot ${state || ""}`;
  }
}

function serverAudioFeature(payload) {
  return {
    rms: typeof payload.input_rms === "number" ? payload.input_rms : 0,
  };
}

function sourceKey(source) {
  return `slot:${source.slot}`;
}

function isLocalPeak(previous, current, next, field, threshold) {
  if (!current || typeof current[field] !== "number" || current[field] < threshold) {
    return false;
  }
  const previousScore = previous && typeof previous[field] === "number" ? previous[field] : Number.NEGATIVE_INFINITY;
  const nextScore = next && typeof next[field] === "number" ? next[field] : Number.NEGATIVE_INFINITY;
  return current[field] >= previousScore && current[field] > nextScore;
}

function clampUnit(value) {
  return Math.max(0, Math.min(1, value || 0));
}

function signalLevelLabel(rms) {
  if (rms <= 0.000001) {
    return "silent";
  }
  return `${(20 * Math.log10(rms)).toFixed(1)} dB`;
}

function latencyLabel(latencyMs) {
  if (typeof latencyMs !== "number") {
    return "-";
  }
  return `${Math.max(0, latencyMs).toFixed(0)} ms`;
}

function colorCss(rgb, level) {
  const alpha = Math.max(0.08, Math.min(1, level));
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
}

new SourceGridApp().start();
