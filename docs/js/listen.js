import { mountChrome } from "./chrome.js?v=listen3";
import { renderDemos } from "./player.js?v=listen3";

mountChrome({ active: "listen" });
renderDemos().catch((err) => console.error(err));
