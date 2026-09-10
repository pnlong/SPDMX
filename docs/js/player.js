async function loadManifest() {
  const res = await fetch("audio/manifest.json");
  if (!res.ok) return { demos: [] };
  return res.json();
}

function stemControls(row) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.gap = "0.35rem";

  const mute = document.createElement("button");
  mute.type = "button";
  mute.textContent = "Mute";
  mute.addEventListener("click", () => {
    const audio = row.querySelector("audio");
    if (!audio) return;
    audio.muted = !audio.muted;
    mute.classList.toggle("active", audio.muted);
    mute.textContent = audio.muted ? "Muted" : "Mute";
  });

  const solo = document.createElement("button");
  solo.type = "button";
  solo.textContent = "Solo";
  solo.addEventListener("click", () => {
    const card = row.closest(".demo-card");
    if (!card) return;
    const audios = [...card.querySelectorAll("audio")];
    const thisAudio = row.querySelector("audio");
    const already = solo.classList.contains("active");
    card.querySelectorAll(".stem-row button").forEach((b) => {
      if (b.textContent === "Solo" || b.textContent.startsWith("Solo")) {
        b.classList.remove("active");
      }
    });
    if (already) {
      audios.forEach((a) => {
        a.muted = false;
      });
      card.querySelectorAll(".stem-row button").forEach((b) => {
        if (b.textContent === "Muted") {
          b.textContent = "Mute";
          b.classList.remove("active");
        }
      });
      return;
    }
    solo.classList.add("active");
    audios.forEach((a) => {
      a.muted = a !== thisAudio;
    });
    card.querySelectorAll(".stem-row").forEach((r) => {
      const m = [...r.querySelectorAll("button")].find((b) =>
        b.textContent.startsWith("Mute")
      );
      const a = r.querySelector("audio");
      if (!m || !a) return;
      m.classList.toggle("active", a.muted);
      m.textContent = a.muted ? "Muted" : "Mute";
    });
  });

  wrap.append(mute, solo);
  return wrap;
}

function buildDemoCard(demo) {
  const card = document.createElement("article");
  card.className = "demo-card";
  card.innerHTML = `
    <header>
      <div>
        <h3>${demo.label}</h3>
        <p class="blurb">${demo.blurb || ""}</p>
      </div>
      <div class="demo-meta">chunk ${demo.chunk} · ${demo.n_tracks} stems</div>
    </header>
  `;
  const list = document.createElement("div");
  list.className = "stem-list";

  if (demo.mix) {
    const row = document.createElement("div");
    row.className = "stem-row mix";
    row.innerHTML = `<strong>Mix</strong><audio controls preload="none" src="${demo.mix}"></audio>`;
    row.append(stemControls(row));
    list.append(row);
  }

  for (const stem of demo.stems || []) {
    const row = document.createElement("div");
    row.className = "stem-row";
    row.innerHTML = `<strong>${stem.label}</strong><audio controls preload="none" src="${stem.file}"></audio>`;
    row.append(stemControls(row));
    list.append(row);
  }

  card.append(list);
  return card;
}

export async function renderDemos() {
  const root = document.getElementById("demo-root");
  if (!root) return;
  const manifest = await loadManifest();
  root.innerHTML = "";
  if (!manifest.demos?.length) {
    root.innerHTML =
      "<p class='chart-note'>Audio demos will appear here after export.</p>";
    return;
  }
  for (const demo of manifest.demos) {
    root.append(buildDemoCard(demo));
  }
}
