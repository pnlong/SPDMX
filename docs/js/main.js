import { renderCharts } from "./charts.js";
import { renderDemos } from "./player.js";

function setupNav() {
  const toggle = document.querySelector(".nav-toggle");
  const nav = document.getElementById("site-nav");
  if (!toggle || !nav) return;

  const setOpen = (open) => {
    nav.classList.toggle("is-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  };

  toggle.addEventListener("click", () => {
    setOpen(!nav.classList.contains("is-open"));
  });

  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setOpen(false));
  });

  window.addEventListener("resize", () => {
    if (window.matchMedia("(min-width: 861px)").matches) setOpen(false);
  });
}

async function boot() {
  setupNav();
  try {
    await renderCharts();
  } catch (err) {
    console.error(err);
  }
  try {
    await renderDemos();
  } catch (err) {
    console.error(err);
  }
}

boot();
