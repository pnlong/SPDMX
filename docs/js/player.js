async function loadManifest() {
  const res = await fetch("audio/manifest.json");
  if (!res.ok) return { demos: [] };
  return res.json();
}

/** Only one demo card may play at a time. */
const mixers = [];

function stopOtherMixers(except) {
  for (const m of mixers) {
    if (m !== except) m.stopAll();
  }
}

/**
 * One Play/Pause/Stop + seek bar; stems play in sync.
 * Mute/Solo only change who is audible. Demos are exclusive.
 */
function attachMixer(card) {
  const stemAudios = [...card.querySelectorAll("audio[data-role='stem']")];
  const playBtn = card.querySelector("[data-action='play']");
  const stopBtn = card.querySelector("[data-action='stop']");
  const timeEl = card.querySelector("[data-role='time']");
  const seek = card.querySelector("[data-role='seek']");
  const rows = [...card.querySelectorAll(".stem-row[data-track]")];

  if (!stemAudios.length || !playBtn || !seek) return;

  let playing = false;
  let seeking = false;
  let raf = 0;

  function fmt(t) {
    if (!Number.isFinite(t)) return "0:00";
    const s = Math.max(0, Math.floor(t));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function duration() {
    const d = Math.max(0, ...stemAudios.map((a) => a.duration || 0));
    return Number.isFinite(d) ? d : 0;
  }

  function currentTime() {
    return stemAudios[0]?.currentTime || 0;
  }

  function seekAll(t) {
    const d = duration();
    const clamped = Math.max(0, Math.min(t, d || t));
    stemAudios.forEach((a) => {
      try {
        a.currentTime = clamped;
      } catch {
        /* not seekable yet */
      }
    });
  }

  function updateProgress() {
    const d = duration();
    const t = currentTime();
    if (timeEl) timeEl.textContent = `${fmt(t)} / ${fmt(d)}`;
    if (!seeking && d > 0) {
      seek.value = String((t / d) * 1000);
      seek.max = "1000";
    }
    seek.disabled = d <= 0;
  }

  function applyMuteSolo() {
    const anySolo = rows.some((r) => r.dataset.solo === "1");
    rows.forEach((row) => {
      const audio = row.querySelector("audio");
      if (!audio) return;
      const muted = row.dataset.mute === "1";
      const solo = row.dataset.solo === "1";
      audio.muted = anySolo ? !solo : muted;
      const muteBtn = row.querySelector("[data-action='mute']");
      const soloBtn = row.querySelector("[data-action='solo']");
      if (muteBtn) {
        muteBtn.classList.toggle("active", muted);
        muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
      }
      if (soloBtn) {
        soloBtn.classList.toggle("active", solo);
        soloBtn.setAttribute("aria-pressed", solo ? "true" : "false");
      }
    });
  }

  function setPlayingUi(on) {
    playing = on;
    playBtn.classList.toggle("is-playing", on);
    playBtn.setAttribute("aria-label", on ? "Pause" : "Play");
    playBtn.setAttribute("aria-pressed", on ? "true" : "false");
    playBtn.title = on ? "Pause" : "Play";
  }

  function pause() {
    stemAudios.forEach((a) => a.pause());
    setPlayingUi(false);
    cancelAnimationFrame(raf);
    updateProgress();
  }

  function stopAll() {
    pause();
    seekAll(0);
    updateProgress();
  }

  const api = { stopAll };
  mixers.push(api);

  async function play() {
    stopOtherMixers(api);
    const t = currentTime();
    await Promise.all(
      stemAudios.map(async (a) => {
        try {
          a.currentTime = t;
          await a.play();
        } catch {
          /* ignore per-stem play failures */
        }
      })
    );
    setPlayingUi(true);
    cancelAnimationFrame(raf);
    const tick = () => {
      updateProgress();
      if (playing) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  playBtn.addEventListener("click", () => {
    if (playing) pause();
    else play();
  });

  stopBtn?.addEventListener("click", stopAll);

  seek.addEventListener("pointerdown", () => {
    seeking = true;
  });
  seek.addEventListener("pointerup", () => {
    seeking = false;
  });
  seek.addEventListener("input", () => {
    const d = duration();
    if (d <= 0) return;
    seekAll((Number(seek.value) / 1000) * d);
    updateProgress();
  });

  stemAudios.forEach((a) => {
    a.addEventListener("ended", () => {
      if (stemAudios.every((x) => x.ended || x.paused)) {
        setPlayingUi(false);
        cancelAnimationFrame(raf);
        if (stemAudios.every((x) => x.ended)) seekAll(0);
        updateProgress();
      }
    });
    a.addEventListener("loadedmetadata", updateProgress);
  });

  rows.forEach((row) => {
    row.querySelector("[data-action='mute']")?.addEventListener("click", () => {
      row.dataset.mute = row.dataset.mute === "1" ? "0" : "1";
      if (row.dataset.mute === "1") row.dataset.solo = "0";
      applyMuteSolo();
    });
    row.querySelector("[data-action='solo']")?.addEventListener("click", () => {
      const on = row.dataset.solo !== "1";
      row.dataset.solo = on ? "1" : "0";
      if (on) row.dataset.mute = "0";
      applyMuteSolo();
    });
  });

  applyMuteSolo();
  updateProgress();
}

function padLabels(labels) {
  const width = Math.max(3, ...labels.map((s) => s.length));
  return { width, padded: labels.map((s) => s.padEnd(width, " ")) };
}

function buildDemoCard(demo) {
  const card = document.createElement("article");
  card.className = "demo-card";

  const stemLabels = (demo.stems || []).map((s) => s.label || `Track ${s.track}`);
  const { width, padded } = padLabels(stemLabels);

  card.innerHTML = `
    <header>
      <div>
        <h3 title="${demo.song_id || ""}">${demo.label}</h3>
        <p class="blurb">${demo.blurb || ""}</p>
      </div>
      <div class="demo-meta">chunk ${demo.chunk} · ${demo.n_tracks} stems</div>
    </header>
    <div class="mixer-transport">
      <button type="button" class="mixer-btn icon-btn primary" data-action="play" aria-label="Play" title="Play">
        <svg class="icon-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
        <svg class="icon-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>
      </button>
      <button type="button" class="mixer-btn icon-btn" data-action="stop" aria-label="Stop" title="Stop">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6h12v12H6z"/></svg>
      </button>
      <span class="mixer-time" data-role="time">0:00 / 0:00</span>
      <input
        class="mixer-seek"
        data-role="seek"
        type="range"
        min="0"
        max="1000"
        value="0"
        step="1"
        aria-label="Seek"
      >
    </div>
  `;

  const list = document.createElement("div");
  list.className = "stem-list";
  list.style.setProperty("--stem-label-ch", String(width));

  (demo.stems || []).forEach((stem, i) => {
    const row = document.createElement("div");
    row.className = "stem-row";
    row.dataset.track = String(stem.track ?? i);
    row.dataset.mute = "0";
    row.dataset.solo = "0";
    row.innerHTML = `
      <span class="stem-label">${padded[i]}</span>
      <div class="stem-actions">
        <button type="button" data-action="mute" aria-pressed="false">Mute</button>
        <button type="button" data-action="solo" aria-pressed="false">Solo</button>
      </div>
      <audio data-role="stem" preload="metadata" src="${stem.file}"></audio>
    `;
    list.append(row);
  });

  card.append(list);
  attachMixer(card);
  return card;
}

export async function renderDemos() {
  const root = document.getElementById("demo-root");
  if (!root) return;
  const manifest = await loadManifest();
  root.innerHTML = "";
  mixers.length = 0;
  if (!manifest.demos?.length) {
    root.innerHTML =
      "<p class='chart-note'>Audio demos will appear here after export.</p>";
    return;
  }
  for (const demo of manifest.demos) {
    root.append(buildDemoCard(demo));
  }
}
