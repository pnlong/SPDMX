import { mountChrome } from "./chrome.js";
import { renderDemos } from "./player.js";

mountChrome({ active: "listen" });
renderDemos().catch((err) => console.error(err));
