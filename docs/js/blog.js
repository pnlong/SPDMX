import { mountChrome } from "../js/chrome.js";

mountChrome({ active: "blog" });

const post = document.querySelector("article.post");
if (post) {
  const back = document.createElement("p");
  back.className = "blog-back";
  back.innerHTML = `<a href="index.html">← Blog</a>`;
  post.prepend(back);
}
