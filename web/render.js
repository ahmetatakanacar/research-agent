let configured = false;

function configure() {
  if (configured) return;
  configured = true;
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A" && node.hasAttribute("href")) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
}

const SANITIZE = {
  FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select", "iframe", "object", "embed", "svg", "math"],
  FORBID_ATTR: ["style", "srcset"],
};

function numberCitations(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement.closest("a, code, pre") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  for (const node of nodes) {
    const text = node.nodeValue;
    if (!/\[\d{1,3}\]/.test(text)) continue;

    const fragment = document.createDocumentFragment();
    let last = 0;
    for (const match of text.matchAll(/\[(\d{1,3})\]/g)) {
      fragment.append(text.slice(last, match.index));
      const link = document.createElement("a");
      link.className = "cite";
      link.href = `#src-${match[1]}`;
      link.dataset.cite = match[1];
      link.textContent = match[1];
      link.removeAttribute("target");
      fragment.append(link);
      last = match.index + match[0].length;
    }
    fragment.append(text.slice(last));
    node.replaceWith(fragment);
  }
}

function tagSources(root, sources) {
  const lists = root.querySelectorAll("ol");
  const list = lists[lists.length - 1];
  if (!list || !sources.length || list.children.length !== sources.length) return;
  [...list.children].forEach((item, index) => {
    item.id = `src-${index + 1}`;
    item.classList.add("source");
  });
}

export function renderReport(target, markdown, sources = []) {
  configure();
  const html = DOMPurify.sanitize(marked.parse(markdown, { gfm: true, breaks: false }), SANITIZE);
  target.innerHTML = html;

  const headings = [];
  target.querySelectorAll("h2").forEach((heading, index) => {
    heading.id = `s${index}`;
    headings.push({ id: heading.id, text: heading.textContent });
  });

  numberCitations(target);
  tagSources(target, sources);

  target.onclick = (event) => {
    const link = event.target.closest("a.cite");
    if (!link) return;
    event.preventDefault();
    const source = target.querySelector(`#src-${link.dataset.cite}`);
    if (!source) return;
    source.scrollIntoView({ behavior: "smooth", block: "center" });
    source.classList.remove("is-flash");
    void source.offsetWidth; // restart the animation
    source.classList.add("is-flash");
  };

  return headings;
}

export function wordCount(markdown) {
  return markdown.split(/\s+/).filter(Boolean).length;
}

export function fileSlug(text) {
  const map = { ç: "c", ğ: "g", ı: "i", ö: "o", ş: "s", ü: "u" };
  return text.toLowerCase().replace(/[çğıöşü]/g, (c) => map[c]).replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "report";
}
