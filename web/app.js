import { ApiError, api, handleUnauthorized, hasApiKey, setApiKey, streamEvents } from "./api.js";
import { applyStatic, count, detectLocale, locale, setLocale, t } from "./i18n.js";
import { fileSlug, renderReport, wordCount } from "./render.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const views = {
  topic: $("#view-topic"),
  team: $("#view-team"),
  run: $("#view-run"),
  report: $("#view-report"),
  history: $("#view-history"),
};
const STEPS = ["topic", "team", "run", "report"];
const WRITE_NODES = ["write_report", "write_introduction", "write_conclusion"];
const MAX_ANALYSTS = 6;
const LANG_KEY = "ra-report-lang";

const state = {};
let currentView = "topic";
let health = null;

function resetState() {
  state.controller?.abort();
  Object.assign(state, {
    threadId: null,
    topic: "",
    language: "en",
    factCheck: true,
    analysts: [],
    interviews: new Map(),
    writeNodes: new Set(),
    reviewDone: false,
    finalized: false,
    report: "",
    sources: [],
    warnings: [],
    reviewNotes: [],
    usage: null,
    createdAt: null,
    lastEventId: 0,
    finished: false,
    controller: null,
  });
}
resetState();

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (node.hidden = true), 3200);
}

function setBusy(button, busy, label) {
  const text = $(".btn__text", button) || button;
  if (busy) {
    button.dataset.label = text.textContent;
    if (label) text.textContent = label;
  } else if (button.dataset.label) {
    text.textContent = button.dataset.label;
    delete button.dataset.label;
  }
  button.disabled = busy;
  button.classList.toggle("is-busy", busy);
}

function errorMessage(error) {
  if (error instanceof ApiError && error.status === 0) return t("net_error");
  return error?.message || t("err_unknown");
}

function setHash(hash) {
  history.replaceState(null, "", hash ? `#${hash}` : location.pathname + location.search);
}

function show(name) {
  currentView = name;
  Object.entries(views).forEach(([key, node]) => (node.hidden = key !== name));
  $("#trail").hidden = !STEPS.includes(name);
  $("#home-btn").toggleAttribute("aria-current", name === "topic");
  $("#history-btn").toggleAttribute("aria-current", name === "history");
  const index = STEPS.indexOf(name);
  $$(".trail li").forEach((li) => {
    const step = STEPS.indexOf(li.dataset.step);
    li.classList.toggle("is-done", step < index);
    li.classList.toggle("is-current", step === index);
    li.toggleAttribute("aria-current", step === index);
  });
  if (show.shown) window.scrollTo({ top: 0, behavior: "smooth" });
  show.shown = true;
}

function formatNumber(n) {
  return new Intl.NumberFormat(locale() === "tr" ? "tr-TR" : "en-US").format(n);
}

function formatTime(seconds) {
  const total = Math.round(seconds);
  return total >= 60 ? `${Math.floor(total / 60)}m ${total % 60}s` : `${total}s`;
}

function usageText(usage) {
  if (!usage || !usage.total_tokens) return "";
  const base = t("usage", { tokens: formatNumber(usage.total_tokens), time: formatTime(usage.elapsed) });
  return usage.cost == null ? base : base + t("usage_cost", { cost: usage.cost.toFixed(2) });
}

function formatDate(seconds) {
  const date = seconds ? new Date(seconds * 1000) : new Date();
  return date.toLocaleDateString(locale() === "tr" ? "tr-TR" : "en-US", { day: "numeric", month: "long", year: "numeric" });
}

let keyPrompt = null;

function promptKey() {
  if (keyPrompt) return keyPrompt;
  keyPrompt = new Promise((resolve) => {
    const dialog = $("#key-dialog");
    const form = $("#key-form");
    const input = $("#key-input");
    const error = $("#key-error");
    error.hidden = true;
    input.value = "";

    dialog.addEventListener("cancel", (event) => event.preventDefault());
    const submit = async (event) => {
      event.preventDefault();
      setApiKey(input.value.trim());
      try {
        await api("/api/auth/check", { noRetry: true });
      } catch (failure) {
        if (failure.status === 401) {
          error.hidden = false;
          input.select();
          return;
        }
      }
      form.removeEventListener("submit", submit);
      dialog.close();
      keyPrompt = null;
      resolve(true);
    };
    form.addEventListener("submit", submit);
    dialog.showModal();
    input.focus();
  });
  return keyPrompt;
}
handleUnauthorized(promptKey);

function renderExamples() {
  const box = $("#examples");
  box.replaceChildren();
  for (const [label, topic] of t("examples")) {
    const chip = el("button", "chip", label);
    chip.type = "button";
    chip.addEventListener("click", () => {
      $("#topic").value = topic;
      $("#topic").focus();
    });
    box.append(chip);
  }
}

function renderHealth() {
  const note = $("#health");
  const key = health && !health.openai ? "health_openai" : health && !health.tavily ? "health_tavily" : null;
  note.hidden = !key;
  if (key) note.textContent = t(key);
}

function fillLanguages() {
  const select = $("#language");
  select.replaceChildren(...health.languages.map(({ code, name }) => {
    const option = el("option", "", name);
    option.value = code;
    return option;
  }));
  let preferred = null;
  try { preferred = localStorage.getItem(LANG_KEY); } catch { /* storage unavailable */ }
  const wanted = preferred || locale();
  select.value = health.languages.some((l) => l.code === wanted) ? wanted : "en";
}

$("#topic-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = $("#topic").value.trim();
  resetState();
  state.topic = topic;
  state.language = $("#language").value;
  try { localStorage.setItem(LANG_KEY, state.language); } catch { /* storage unavailable */ }

  const button = $("#start-btn");
  setBusy(button, true, t("start_busy"));
  try {
    const { thread_id: threadId } = await api("/api/research", {
      method: "POST",
      body: {
        topic,
        max_analyst: Number(document.querySelector('input[name="count"]:checked').value),
        language: state.language,
      },
    });
    state.threadId = threadId;
    setHash(`/r/${threadId}`);
    await follow(threadId);
  } catch (error) {
    toast(errorMessage(error));
  } finally {
    setBusy(button, false);
  }
});

async function follow(threadId) {
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.finished = false;
  let failures = 0;

  while (!state.finished && !controller.signal.aborted) {
    try {
      await streamEvents(`/api/research/${threadId}/events`, {
        after: state.lastEventId,
        signal: controller.signal,
        onEvent: (event, data, id) => {
          if (id !== null) state.lastEventId = id;
          handleEvent(event, data);
        },
      });
      failures = 0;
      if (!state.finished) await sleep(800);
    } catch (error) {
      if (error.name === "AbortError") return;
      if (error.status === 404) {
        await restore(threadId);
        return;
      }
      if (++failures > 5) {
        showFailure({ message: t("net_error") });
        return;
      }
      await sleep(1000 * failures);
    }
  }
}

function startSegment() {
  state.lastEventId = 0;
  state.finished = false;
}

function handleEvent(event, data) {
  switch (event) {
    case "analysts":
      state.analysts = data.analysts.map((analyst) => ({ ...analyst }));
      renderTeam();
      show("team");
      break;
    case "interview_start":
      addInterview(data.interview, data.analyst, data.role);
      break;
    case "message":
      addBubble(data.interview, data.role, data.content);
      break;
    case "search":
      addTrace(data);
      break;
    case "interview_step":
      if (data.node === "write_section") finishInterview(data.interview);
      break;
    case "stage":
      onStage(data.node);
      break;
    case "draft":
      appendDraft(data.delta);
      break;
    case "usage":
      state.usage = data;
      $("#run-usage").textContent = usageText(data);
      break;
    case "report":
      state.report = data.final_report;
      state.sources = data.sources || [];
      state.warnings = data.warnings || [];
      state.reviewNotes = data.review_notes || [];
      state.usage = data.usage || state.usage;
      state.createdAt = null;
      renderVine();
      showReport();
      break;
    case "error":
      showFailure(data);
      break;
    case "cancelled":
      showFailure({ message: t("stopped_msg") });
      break;
    case "done":
      state.finished = true;
      break;
  }
}

function onStage(node) {
  if (WRITE_NODES.includes(node)) state.writeNodes.add(node);
  else if (node === "review_report") state.reviewDone = true;
  else if (node === "finalize_report") state.finalized = true;
  renderVine();
}

function initials(name) {
  return (name || "").split(/\s+/).filter(Boolean).slice(0, 2).map((word) => word[0].toUpperCase()).join("") || "?";
}

function labeled(labelKey, control) {
  const wrapper = el("label", "mini");
  wrapper.append(el("span", "mini__label", t(labelKey)), control);
  return wrapper;
}

function textField(analyst, key, maxLength, multiline = false) {
  const control = document.createElement(multiline ? "textarea" : "input");
  if (multiline) control.rows = 3;
  else control.type = "text";
  control.maxLength = maxLength;
  control.value = analyst[key] ?? "";
  control.addEventListener("input", () => {
    analyst[key] = control.value;
    control.classList.remove("is-invalid");
  });
  return control;
}

function analystCard(analyst, index) {
  const card = el("article", "analyst card");
  card.style.setProperty("--i", index);

  const avatar = el("span", "avatar", initials(analyst.name));
  const name = textField(analyst, "name", 120);
  name.classList.add("analyst__name");
  name.addEventListener("input", () => (avatar.textContent = initials(name.value)));
  const role = textField(analyst, "role", 200);

  const remove = el("button", "icon-btn", "×");
  remove.type = "button";
  remove.setAttribute("aria-label", t("analyst_remove"));
  remove.title = t("analyst_remove");
  remove.disabled = state.analysts.length <= 1;
  remove.addEventListener("click", () => {
    state.analysts.splice(index, 1);
    renderTeam({ keepFeedback: true });
  });

  const head = el("div", "analyst__head");
  const fields = el("div", "analyst__fields");
  fields.append(labeled("analyst_name", name), labeled("analyst_role", role));
  head.append(avatar, fields, remove);

  card.append(
    head,
    labeled("analyst_aff", textField(analyst, "affiliation", 200)),
    labeled("analyst_desc", textField(analyst, "description", 1500, true)),
  );
  return card;
}

function renderTeam({ keepFeedback = false } = {}) {
  const list = $("#analysts");
  list.classList.toggle("is-static", keepFeedback);
  list.replaceChildren(...state.analysts.map(analystCard));
  $("#add-analyst").hidden = state.analysts.length >= MAX_ANALYSTS;
  if (!keepFeedback) $("#feedback").value = "";
}

$("#add-analyst").addEventListener("click", () => {
  if (state.analysts.length >= MAX_ANALYSTS) return;
  state.analysts.push({ name: "", role: "", affiliation: "", description: "" });
  renderTeam({ keepFeedback: true });
  $$(".analyst__name").at(-1)?.focus();
});

function collectTeam() {
  let valid = true;
  const cards = $$("#analysts .analyst");
  state.analysts.forEach((analyst, index) => {
    for (const [key, selector] of [["name", ".analyst__name"], ["role", ".analyst__fields label:nth-child(2) input"]]) {
      const empty = !analyst[key]?.trim();
      cards[index]?.querySelector(selector)?.classList.toggle("is-invalid", empty);
      if (empty) valid = false;
    }
  });
  if (!valid) {
    toast(t("team_invalid"));
    $("#analysts .is-invalid")?.focus();
    return null;
  }
  return state.analysts.map(({ name, role, affiliation, description }) => ({
    name: name.trim(), role: role.trim(), affiliation: (affiliation || "").trim(), description: (description || "").trim(),
  }));
}

async function sendFeedback(feedback, team) {
  await api(`/api/research/${state.threadId}/feedback`, { method: "POST", body: { feedback, analysts: team } });
  startSegment();
}

$("#feedback-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const team = collectTeam();
  if (!team) return;

  const buttons = [$("#regen-btn"), $("#approve-btn")];
  buttons.forEach((button) => (button.disabled = true));
  try {
    await sendFeedback("", team);
  } catch (error) {
    toast(errorMessage(error));
    buttons.forEach((button) => (button.disabled = false));
    return;
  }
  buttons.forEach((button) => (button.disabled = false));
  beginRun();
  follow(state.threadId);
});

$("#regen-btn").addEventListener("click", async () => {
  const feedback = $("#feedback").value.trim();
  if (!feedback) {
    toast(t("regen_need"));
    $("#feedback").focus();
    return;
  }
  const team = collectTeam();
  if (!team) return;

  const buttons = [$("#regen-btn"), $("#approve-btn")];
  buttons.forEach((button) => (button.disabled = true));
  const label = $("#regen-btn").textContent;
  $("#regen-btn").textContent = t("regen_busy");
  try {
    await sendFeedback(feedback, team);
    await follow(state.threadId);
  } catch (error) {
    toast(errorMessage(error));
  } finally {
    $("#regen-btn").textContent = label;
    buttons.forEach((button) => (button.disabled = false));
  }
});

function beginRun() {
  state.interviews.clear();
  state.writeNodes.clear();
  state.reviewDone = false;
  state.finalized = false;
  $("#interviews").replaceChildren();
  $("#draft").textContent = "";
  $("#draft-panel").hidden = true;
  $("#run-topic").textContent = state.topic;
  $("#run-usage").textContent = usageText(state.usage);
  $("#run-error").hidden = true;
  $("#stop-btn").hidden = false;
  $("#h-run").textContent = t("run_title");
  renderVine();
  show("run");
}

function renderVine() {
  const total = state.analysts.length;
  const interviewsDone = [...state.interviews.values()].filter((i) => i.done).length;
  const allInterviews = total > 0 && interviewsDone >= total;
  const writingDone = state.writeNodes.size >= WRITE_NODES.length;
  const reviewDone = !state.factCheck || state.reviewDone;
  const reportDone = Boolean(state.report);

  const stages = [
    { title: t("stage_team"), meta: count(total, "analyst_one", "analyst_many"), status: "done" },
    {
      title: t("stage_interviews"),
      meta: t("meta_complete", { done: Math.min(interviewsDone, total), total }),
      status: allInterviews ? "done" : "active",
    },
    {
      title: t("stage_writing"),
      meta: t("meta_writing", { n: state.writeNodes.size }),
      status: writingDone ? "done" : allInterviews ? "active" : "idle",
    },
  ];
  if (state.factCheck) {
    stages.push({
      title: t("stage_review"),
      meta: "",
      status: reviewDone && writingDone ? "done" : writingDone ? "active" : "idle",
    });
  }
  stages.push({
    title: t("stage_assemble"),
    meta: "",
    status: reportDone ? "done" : writingDone && reviewDone ? "active" : "idle",
  });

  const check = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 7.5 5.8 10 11 4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const vine = $("#vine");
  vine.replaceChildren(...stages.map((stage) => {
    const item = el("li", `stage stage--${stage.status}`);
    const node = el("span", "stage__node");
    node.innerHTML = check; // constant markup, no user text
    const body = el("div");
    body.append(el("div", "stage__title", stage.title), el("div", "stage__meta", stage.meta));
    item.append(node, body);
    return item;
  }));

  const done = stages.filter((stage) => stage.status === "done").length;
  vine.style.setProperty("--p", Math.min(done, stages.length - 1) / (stages.length - 1));
  $("#stop-btn").hidden = reportDone;
}

function errorText(error) {
  const known = error.code && t(`err_${error.code}`) !== `err_${error.code}`;
  const text = known ? t(`err_${error.code}`) : error.message || t("err_unknown");
  return error.ref ? `${text} ${t("err_ref", { ref: error.ref })}` : text;
}

function markInterviewsStopped() {
  for (const interview of state.interviews.values()) {
    if (interview.done) continue;
    interview.card.classList.replace("is-live", "is-stopped");
    interview.badge.textContent = t("badge_stopped");
  }
}

function showFailure(error) {
  markInterviewsStopped();
  $("#run-error-text").textContent = errorText(error);
  $("#run-error").hidden = false;
  $("#retry-btn").hidden = error.code === "auth";
  $("#h-run").textContent = t("run_stopped_title");
  $("#stop-btn").hidden = true;
  $(".stage--active", $("#vine"))?.classList.remove("stage--active");
  if (currentView !== "run") {
    beginRun();
    $("#run-error").hidden = false;
    $("#stop-btn").hidden = true;
    $("#h-run").textContent = t("run_stopped_title");
  }
}

function sourceName(source) {
  return t(`source_${source}`);
}

function addInterview(id, analyst, role) {
  if (state.interviews.has(id)) return;

  const card = el("article", "interview card is-live");
  const head = el("div", "interview__head");
  const title = el("div", "interview__title");
  title.append(el("strong", "", analyst || t("interview_n", { n: id })));
  if (role) title.append(el("small", "", role));
  head.append(title, el("span", "badge", t("badge_live")));
  const transcript = el("div", "transcript");
  card.append(head, transcript);

  $("#interviews").append(card);
  state.interviews.set(id, { card, transcript, badge: $(".badge", head), done: false });
  renderVine();
}

function scrollDown(interview) {
  interview.transcript.scrollTop = interview.transcript.scrollHeight;
}

function addBubble(id, role, content) {
  const interview = state.interviews.get(id);
  if (!interview || !content) return;

  const bubble = el("div", `bubble bubble--${role}`);
  bubble.append(el("small", "", role === "expert" ? t("role_expert") : t("role_analyst")), document.createTextNode(content));
  interview.transcript.append(bubble);
  scrollDown(interview);
}

function addTrace({ interview: id, source, found, titles }) {
  const interview = state.interviews.get(id);
  if (!interview) return;

  const name = sourceName(source);
  const trace = el("div", found ? "trace" : "trace trace--miss", found ? t("trace_found", { source: name, n: found }) : t("trace_none", { source: name }));
  if (titles?.length) trace.title = titles.join("\n");
  interview.transcript.append(trace);
  scrollDown(interview);
}

function finishInterview(id) {
  const interview = state.interviews.get(id);
  if (!interview || interview.done) return;
  interview.done = true;
  interview.card.classList.replace("is-live", "is-done");
  interview.badge.textContent = t("badge_done");
  renderVine();
}

function appendDraft(delta) {
  const panel = $("#draft-panel");
  const box = $("#draft");
  panel.hidden = false;
  const nearEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.textContent += delta;
  if (nearEnd) box.scrollTop = box.scrollHeight;
}

$("#stop-btn").addEventListener("click", async () => {
  try {
    await api(`/api/research/${state.threadId}/cancel`, { method: "POST" });
  } catch (error) {
    toast(errorMessage(error));
  }
});

$("#retry-btn").addEventListener("click", async () => {
  try {
    await api(`/api/research/${state.threadId}/resume`, { method: "POST" });
  } catch (error) {
    toast(errorMessage(error));
    return;
  }
  startSegment();
  beginRun();
  follow(state.threadId);
});

$("#error-back").addEventListener("click", newResearch);

function showReport() {
  $("#draft-panel").hidden = true;
  const headings = renderReport($("#report-body"), state.report, state.sources);

  $("#toc").replaceChildren(...headings.map(({ id, text }) => {
    const item = el("li");
    const link = el("a", "", text);
    link.href = `#${id}`;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    });
    item.append(link);
    return item;
  }));
  $(".toc").hidden = headings.length === 0;
  renderReportMeta();
  renderReportNotes();
  show("report");
}

function renderReportMeta() {
  const words = wordCount(state.report);
  const meta = t("report_meta", {
    topic: state.topic,
    analysts: count(state.analysts.length, "analyst_one", "analyst_many"),
    words,
    minutes: Math.max(1, Math.round(words / 200)),
    date: formatDate(state.createdAt),
  });
  $("#report-meta").textContent = [meta, usageText(state.usage)].filter(Boolean).join(" · ");
}

const WARNING_TEXT = {
  no_sources: "warn_no_sources",
  review_failed: "warn_review_failed",
  search_credits_exhausted: "warn_search_credits",
};

function renderReportNotes() {
  const warnings = $("#report-warnings");
  warnings.replaceChildren(...state.warnings.map((warning) => {
    const key = WARNING_TEXT[warning.code];
    return el("li", "", key ? t(key, { analyst: warning.analyst || "" }) : String(warning.code));
  }));
  warnings.hidden = state.warnings.length === 0;

  const panel = $("#review-panel");
  panel.hidden = !state.factCheck;
  if (!state.factCheck) return;
  const notes = state.reviewNotes;
  $("#review-summary").textContent = notes.length ? t("review_summary", { n: notes.length }) : t("review_none");
  $("#review-list").replaceChildren(...notes.map((note) => {
    const item = el("li");
    item.append(
      el("strong", "", note.claim),
      document.createTextNode(` — ${note.problem} `),
      el("em", "", `(${note.fixed ? t("review_fixed") : t("review_unfixed")})`),
    );
    return item;
  }));
  panel.open = notes.length > 0;
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("#copy-btn").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(state.report);
    toast(t("copied"));
  } catch {
    toast(t("copy_denied"));
  }
});

$("#download-btn").addEventListener("click", () => {
  saveBlob(new Blob([state.report], { type: "text/markdown;charset=utf-8" }), `${fileSlug(state.topic)}.md`);
});

$("#print-btn").addEventListener("click", () => window.print());

$("#again-btn").addEventListener("click", newResearch);

function newResearch() {
  resetState();
  setHash("");
  $("#topic").value = "";
  show("topic");
  $("#topic").focus();
}

function goHome() {
  if (currentView === "run" && !state.finished) toast(t("home_running"));
  if (state.threadId) {
    newResearch();
  } else {
    setHash("");
    show("topic");
  }
}

$("#home-btn").addEventListener("click", goHome);

async function openHistory() {
  state.controller?.abort();
  let runs;
  try {
    ({ runs } = await api("/api/research"));
  } catch (error) {
    toast(errorMessage(error));
    return;
  }
  setHash("/history");
  renderHistory(runs);
  show("history");
}

function renderHistory(runs) {
  const list = $("#history-list");
  list.replaceChildren(...runs.map((run) => {
    const item = el("li", "card history__item");
    const info = el("div", "history__info");
    const meta = el("div", "history__meta");
    meta.append(
      el("span", `chip chip--${run.status}`, t(`status_${run.status}`)),
      el("span", "", formatDate(run.created_at)),
      el("span", "", run.language.toUpperCase()),
    );
    info.append(el("strong", "history__topic", run.topic), meta);

    const open = el("button", "btn btn--primary btn--small", t("history_open"));
    open.type = "button";
    open.addEventListener("click", () => {
      setHash(`/r/${run.thread_id}`);
      openRun(run.thread_id);
    });
    const remove = el("button", "btn btn--ghost btn--small", t("history_delete"));
    remove.type = "button";
    remove.addEventListener("click", async () => {
      if (!confirm(t("confirm_delete"))) return;
      try {
        await api(`/api/research/${run.thread_id}`, { method: "DELETE" });
        item.remove();
        $("#history-empty").hidden = list.children.length > 0;
      } catch (error) {
        toast(errorMessage(error));
      }
    });
    const actions = el("div", "actions");
    actions.append(open, remove);
    item.append(info, actions);
    return item;
  }));
  $("#history-empty").hidden = runs.length > 0;
}

$("#history-btn").addEventListener("click", openHistory);

async function restore(threadId) {
  let run;
  try {
    run = await api(`/api/research/${threadId}`);
  } catch (error) {
    resetState();
    setHash("");
    toast(error.status === 404 ? t("not_found") : errorMessage(error));
    show("topic");
    return;
  }

  resetState();
  Object.assign(state, {
    threadId,
    topic: run.topic,
    language: run.language,
    factCheck: run.settings?.fact_check !== false,
    analysts: (run.analysts || []).map((analyst) => ({ ...analyst })),
    usage: run.usage,
    sources: run.sources,
    warnings: run.warnings,
    reviewNotes: run.review_notes,
    createdAt: run.created_at,
  });

  switch (run.status) {
    case "awaiting_approval":
      renderTeam();
      show("team");
      break;
    case "running":
      beginRun();
      follow(threadId);
      break;
    case "done":
      state.report = run.report;
      showReport();
      break;
    default:
      beginRun();
      showFailure(run.error || { message: t("stopped_msg") });
  }
}

async function openRun(threadId) {
  await restore(threadId);
}

function refreshLanguageButtons() {
  $$("[data-ui-lang]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.uiLang === locale())));
}

function rerender() {
  applyStatic();
  refreshLanguageButtons();
  renderExamples();
  renderHealth();
  if (currentView === "team") renderTeam({ keepFeedback: true });
  if (currentView === "run") {
    renderVine();
    $("#run-usage").textContent = usageText(state.usage);
  }
  if (currentView === "report") {
    renderReportMeta();
    renderReportNotes();
  }
  if (currentView === "history") openHistory();
}

$$("[data-ui-lang]").forEach((button) => button.addEventListener("click", () => {
  setLocale(button.dataset.uiLang);
  rerender();
}));

async function init() {
  setLocale(detectLocale(), { persist: false });
  refreshLanguageButtons();
  renderExamples();

  try {
    health = await api("/api/health");
  } catch {
    health = { ok: false, openai: true, tavily: true, auth: false, languages: [{ code: "en", name: "English" }] };
    toast(t("net_error"));
  }
  fillLanguages();
  renderHealth();

  if (health.auth && !hasApiKey()) await promptKey();

  const match = location.hash.match(/^#\/r\/([0-9a-f]{32})$/);
  if (match) {
    await openRun(match[1]);
  } else if (location.hash === "#/history") {
    await openHistory();
  } else {
    show("topic");
  }
}

init();
