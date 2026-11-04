import { mountChrome } from "./chrome.js";

mountChrome({ active: "download" });

function copyText(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    return Promise.resolve();
  } finally {
    document.body.removeChild(ta);
  }
}

function setupCopyButtons() {
  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-copy-target");
      const block = id ? document.getElementById(id) : null;
      if (!block) return;
      const text = block.textContent ?? "";
      try {
        await copyText(text);
        const prev = btn.textContent;
        btn.textContent = "Copied";
        btn.classList.add("is-copied");
        window.setTimeout(() => {
          btn.textContent = prev;
          btn.classList.remove("is-copied");
        }, 1600);
      } catch (err) {
        console.error(err);
        btn.textContent = "Failed";
        window.setTimeout(() => {
          btn.textContent = "Copy";
        }, 1600);
      }
    });
  });
}

setupCopyButtons();
