/** Shared header/nav for SPDMX project pages. */
const NAV = [
  { href: "index.html", id: "home", label: "Home" },
  { href: "about.html", id: "about", label: "About" },
  { href: "listen.html", id: "listen", label: "Listen" },
  { href: "soundfonts.html", id: "soundfonts", label: "Soundfonts" },
  { href: "blog/index.html", id: "blog", label: "Blog" },
];

function pageDepth() {
  const path = window.location.pathname.replace(/\/+$/, "");
  if (path.includes("/blog/")) return "../";
  return "";
}

function currentTab() {
  const path = window.location.pathname;
  if (path.includes("/blog")) return "blog";
  if (path.endsWith("about.html")) return "about";
  if (path.endsWith("listen.html")) return "listen";
  if (path.endsWith("soundfonts.html")) return "soundfonts";
  return "home";
}

export function mountChrome({ active } = {}) {
  const prefix = pageDepth();
  const tab = active || currentTab();
  const header = document.querySelector(".site-header");
  if (!header) return;

  const brandHref = `${prefix}index.html`;
  const logoSrc = `${prefix}assets/logo.png`;

  header.innerHTML = `
    <a class="brand" href="${brandHref}">
      <img class="brand-mark" src="${logoSrc}" alt="SPDMX" width="40" height="40" onerror="this.style.display='none'">
      <span class="brand-name">SPDMX</span>
    </a>
    <button
      class="nav-toggle"
      type="button"
      aria-label="Open menu"
      aria-controls="site-nav"
      aria-expanded="false"
    >
      <span class="nav-toggle-bar" aria-hidden="true"></span>
      <span class="nav-toggle-bar" aria-hidden="true"></span>
      <span class="nav-toggle-bar" aria-hidden="true"></span>
    </button>
    <nav class="nav" id="site-nav"></nav>
  `;

  const nav = header.querySelector("#site-nav");
  for (const item of NAV) {
    const a = document.createElement("a");
    a.href = `${prefix}${item.href}`;
    a.textContent = item.label;
    if (item.id === tab) {
      a.classList.add("is-active");
      a.setAttribute("aria-current", "page");
    }
    nav.appendChild(a);
  }

  const footer = document.querySelector(".site-footer");
  if (footer && !footer.dataset.mounted) {
    footer.dataset.mounted = "1";
    footer.innerHTML = `<p>SPDMX · <a href="https://github.com/pnlong/SPDMX">GitHub</a> · <a href="${prefix}blog/index.html">Blog</a></p>`;
  }

  setupNav();
}

export function setupNav() {
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
