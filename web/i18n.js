const en = {
  nav_home: "Home",
  nav_history: "History",
  home_running: "The research keeps running on the server. Find it under History.",
  trail_topic: "Topic", trail_team: "Team", trail_run: "Interviews", trail_report: "Report",

  hero_eyebrow: "Multi-agent research",
  hero_title: "Plant a topic.<br>Let <em>a team</em> research it.",
  hero_lede: "Research Agent assembles expert analysts for your topic, interviews each one using web and Wikipedia search, and writes a single sourced report.",
  topic_label: "Research topic",
  topic_placeholder: "e.g. How urban forests improve city climate resilience",
  examples: [
    ["Sustainable farming & AI", "The role of AI in sustainable agriculture"],
    ["Multi-agent systems", "How to design multi-agent systems with LangGraph"],
    ["Urban forests", "The impact of urban forests on climate resilience"],
  ],
  language_label: "Report language",
  analysts_label: "Number of analysts",
  analysts_hint: "More analysts means broader perspectives, but also a longer run and higher cost.",
  start_btn: "Assemble team", start_busy: "Assembling team",

  team_eyebrow: "Step 2 · Approval",
  team_title: "Your team is ready",
  team_lede: "Edit the analysts, remove or add one, or describe the changes you want and regenerate the team.",
  analyst_name: "Name", analyst_role: "Role", analyst_aff: "Affiliation", analyst_desc: "Focus and motives",
  analyst_remove: "Remove analyst", analyst_add: "Add analyst",
  analyst_new: "New analyst",
  feedback_label: "Requested changes",
  optional: "(optional)",
  feedback_placeholder: "e.g. Add an economist and a field engineer.",
  regen_btn: "Regenerate team", regen_busy: "Regenerating…",
  approve_btn: "Approve & start",
  team_invalid: "Every analyst needs a name and a role.",
  regen_need: "Describe what you want changed to regenerate the team.",

  run_eyebrow: "Step 3 · Live",
  run_title: "Research in progress",
  run_stopped_title: "Research stopped",
  stage_team: "Analyst team assembled",
  stage_interviews: "Expert interviews",
  stage_writing: "Writing the report",
  stage_review: "Fact-checking",
  stage_assemble: "Assembling report",
  meta_complete: "{done}/{total} complete",
  meta_writing: "{n}/3 · body, intro, conclusion",
  analyst_one: "{n} analyst", analyst_many: "{n} analysts",
  interview_n: "Interview {n}",
  badge_live: "live", badge_done: "done", badge_stopped: "stopped",
  role_expert: "Expert", role_analyst: "Analyst",
  trace_found: "{source} · {n} found",
  trace_none: "{source} · no results",
  source_web: "web search", source_wikipedia: "Wikipedia",
  draft_title: "Live draft",
  stop_btn: "Stop",
  retry_btn: "Try again",
  start_over: "Start over",
  stopped_msg: "The research was stopped. You can continue it from where it left off.",
  err_auth: "The OpenAI API key was rejected.",
  err_rate_limit: "The OpenAI API rate limit or quota was reached.",
  err_model: "The configured OpenAI model was not found.",
  err_timeout: "The model took too long to respond.",
  err_connection: "Could not reach the model provider.",
  err_unknown: "Something went wrong while researching.",
  err_ref: "Reference: {ref}",
  net_error: "Could not reach the server.",
  usage: "{tokens} tokens · {time}",
  usage_cost: " · ≈ ${cost}",

  report_eyebrow: "Step 4 · Complete",
  report_title: "Your report is ready",
  contents: "Contents",
  copy_md: "Copy Markdown",
  download_md: "Download .md",
  print: "Print / PDF",
  new_research: "New research",
  copied: "Report copied to clipboard.",
  copy_denied: "Could not copy; the browser denied clipboard access.",
  review_summary: "Fact-check: {n} claims reviewed",
  review_none: "Fact-check: every checked claim is supported by its sources.",
  review_fixed: "corrected", review_unfixed: "could not be corrected",
  warn_no_sources: "No sources were found for {analyst}’s focus area, so that part of the report is empty.",
  warn_review_failed: "The fact-check could not be completed, so the report has not been reviewed.",
  warn_search_credits: "The Tavily search credits ran out during {analyst}’s interview, so that part relies on Wikipedia only.",
  report_meta: "{topic} · {analysts} · ~{words} words · {minutes} min read · {date}",

  history_title: "Past research",
  history_lede: "Everything you researched on this server. Open a report, or delete what you no longer need.",
  history_empty: "Nothing here yet. Start your first research.",
  history_open: "Open", history_delete: "Delete",
  confirm_delete: "Delete this research and its report?",
  status_done: "Done", status_running: "Running", status_awaiting_approval: "Waiting for approval",
  status_error: "Failed", status_cancelled: "Stopped", status_interrupted: "Interrupted",
  not_found: "This research no longer exists.",

  key_title: "API key required",
  key_text: "This server is protected. Enter its API key to continue.",
  key_label: "API key",
  key_save: "Continue",
  key_wrong: "That key was not accepted.",
  health_openai: "OPENAI_API_KEY is not set — check your .env file.",
  health_tavily: "TAVILY_API_KEY is not set — web search will not work.",
  footer: "Research Agent is an AI system; verify the information in reports against their sources.",
  noscript: "This app requires JavaScript.",
  lang_switch: "Interface language",
};

const tr = {
  nav_home: "Ana sayfa",
  nav_history: "Geçmiş",
  home_running: "Araştırma sunucuda çalışmaya devam ediyor. Geçmiş sayfasından bulabilirsin.",
  trail_topic: "Konu", trail_team: "Ekip", trail_run: "Röportajlar", trail_report: "Rapor",

  hero_eyebrow: "Çok ajanlı araştırma",
  hero_title: "Bir konu ek.<br><em>Bir ekip</em> araştırsın.",
  hero_lede: "Research Agent konun için uzman analistler kurar, her biriyle web ve Wikipedia aramalarını kullanarak röportaj yapar ve kaynaklı tek bir rapor yazar.",
  topic_label: "Araştırma konusu",
  topic_placeholder: "örn. Kentsel ormanlar kentlerin iklim dayanıklılığını nasıl artırır",
  examples: [
    ["Sürdürülebilir tarım ve yapay zekâ", "Yapay zekânın sürdürülebilir tarımdaki rolü"],
    ["Çok ajanlı sistemler", "LangGraph ile çok ajanlı sistemler nasıl tasarlanır"],
    ["Kentsel ormanlar", "Kentsel ormanların iklim dayanıklılığına etkisi"],
  ],
  language_label: "Rapor dili",
  analysts_label: "Analist sayısı",
  analysts_hint: "Daha fazla analist daha geniş bakış açıları demektir, ama araştırma uzar ve maliyet artar.",
  start_btn: "Ekibi oluştur", start_busy: "Ekip kuruluyor",

  team_eyebrow: "Adım 2 · Onay",
  team_title: "Ekibin hazır",
  team_lede: "Analistleri düzenle, çıkar ya da ekle; ya da istediğin değişiklikleri yazıp ekibi yeniden oluştur.",
  analyst_name: "Ad", analyst_role: "Rol", analyst_aff: "Kurum", analyst_desc: "Odak ve motivasyon",
  analyst_remove: "Analisti çıkar", analyst_add: "Analist ekle",
  analyst_new: "Yeni analist",
  feedback_label: "İstenen değişiklikler",
  optional: "(isteğe bağlı)",
  feedback_placeholder: "örn. Bir ekonomist ve bir saha mühendisi ekle.",
  regen_btn: "Ekibi yeniden oluştur", regen_busy: "Yeniden oluşturuluyor…",
  approve_btn: "Onayla ve başlat",
  team_invalid: "Her analistin bir adı ve rolü olmalı.",
  regen_need: "Ekibi yeniden oluşturmak için neyin değişmesini istediğini yaz.",

  run_eyebrow: "Adım 3 · Canlı",
  run_title: "Araştırma sürüyor",
  run_stopped_title: "Araştırma durdu",
  stage_team: "Analist ekibi kuruldu",
  stage_interviews: "Uzman röportajları",
  stage_writing: "Rapor yazılıyor",
  stage_review: "Doğrulama",
  stage_assemble: "Rapor birleştiriliyor",
  meta_complete: "{done}/{total} tamamlandı",
  meta_writing: "{n}/3 · gövde, giriş, sonuç",
  analyst_one: "{n} analist", analyst_many: "{n} analist",
  interview_n: "Röportaj {n}",
  badge_live: "canlı", badge_done: "bitti", badge_stopped: "durdu",
  role_expert: "Uzman", role_analyst: "Analist",
  trace_found: "{source} · {n} sonuç",
  trace_none: "{source} · sonuç yok",
  source_web: "web araması", source_wikipedia: "Wikipedia",
  draft_title: "Canlı taslak",
  stop_btn: "Durdur",
  retry_btn: "Tekrar dene",
  start_over: "Baştan başla",
  stopped_msg: "Araştırma durduruldu. Kaldığı yerden devam ettirebilirsin.",
  err_auth: "OpenAI API anahtarı reddedildi.",
  err_rate_limit: "OpenAI API hız sınırına ya da kotasına ulaşıldı.",
  err_model: "Ayarlanan OpenAI modeli bulunamadı.",
  err_timeout: "Model yanıt vermekte çok gecikti.",
  err_connection: "Model sağlayıcısına ulaşılamadı.",
  err_unknown: "Araştırma sırasında bir şeyler ters gitti.",
  err_ref: "Referans: {ref}",
  net_error: "Sunucuya ulaşılamadı.",
  usage: "{tokens} token · {time}",
  usage_cost: " · ≈ ${cost}",

  report_eyebrow: "Adım 4 · Tamamlandı",
  report_title: "Raporun hazır",
  contents: "İçindekiler",
  copy_md: "Markdown kopyala",
  download_md: ".md indir",
  print: "Yazdır / PDF",
  new_research: "Yeni araştırma",
  copied: "Rapor panoya kopyalandı.",
  copy_denied: "Kopyalanamadı; tarayıcı pano erişimini reddetti.",
  review_summary: "Doğrulama: {n} iddia incelendi",
  review_none: "Doğrulama: incelenen her iddia kaynaklarınca destekleniyor.",
  review_fixed: "düzeltildi", review_unfixed: "düzeltilemedi",
  warn_no_sources: "{analyst} için odak alanında kaynak bulunamadı, bu yüzden raporun o bölümü boş.",
  warn_review_failed: "Doğrulama tamamlanamadı, bu yüzden rapor denetlenmedi.",
  warn_search_credits: "{analyst} için yapılan röportajda Tavily arama kredisi bitti, bu yüzden o bölüm yalnızca Wikipedia’ya dayanıyor.",
  report_meta: "{topic} · {analysts} · ~{words} kelime · {minutes} dk okuma · {date}",

  history_title: "Geçmiş araştırmalar",
  history_lede: "Bu sunucuda yaptığın tüm araştırmalar. Bir raporu aç ya da artık gerekmeyenleri sil.",
  history_empty: "Henüz bir şey yok. İlk araştırmanı başlat.",
  history_open: "Aç", history_delete: "Sil",
  confirm_delete: "Bu araştırma ve raporu silinsin mi?",
  status_done: "Bitti", status_running: "Sürüyor", status_awaiting_approval: "Onay bekliyor",
  status_error: "Başarısız", status_cancelled: "Durduruldu", status_interrupted: "Yarıda kaldı",
  not_found: "Bu araştırma artık yok.",

  key_title: "API anahtarı gerekli",
  key_text: "Bu sunucu korumalı. Devam etmek için API anahtarını gir.",
  key_label: "API anahtarı",
  key_save: "Devam",
  key_wrong: "Bu anahtar kabul edilmedi.",
  health_openai: "OPENAI_API_KEY ayarlı değil — .env dosyanı kontrol et.",
  health_tavily: "TAVILY_API_KEY ayarlı değil — web araması çalışmayacak.",
  footer: "Research Agent bir yapay zekâ sistemidir; raporlardaki bilgileri kaynaklarıyla karşılaştırarak doğrula.",
  noscript: "Bu uygulama JavaScript gerektirir.",
  lang_switch: "Arayüz dili",
};

export const LOCALES = { en, tr };
const STORAGE_KEY = "ra-ui-lang";

let current = "en";

function readStored() {
  try { return localStorage.getItem(STORAGE_KEY); } catch { return null; }
}

export function detectLocale() {
  const stored = readStored();
  if (stored && LOCALES[stored]) return stored;
  return (navigator.language || "en").toLowerCase().startsWith("tr") ? "tr" : "en";
}

export function locale() { return current; }

export function t(key, vars) {
  const value = LOCALES[current][key] ?? LOCALES.en[key] ?? key;
  if (typeof value !== "string" || !vars) return value;
  return value.replace(/\{(\w+)\}/g, (_, name) => String(vars[name] ?? ""));
}

export function count(n, oneKey, manyKey) {
  return t(n === 1 ? oneKey : manyKey, { n });
}

// Static markup is translated through data attributes; strings with markup go through data-i18n-html
// (those come from this file only, never from users or the model).
export function applyStatic(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  root.querySelectorAll("[data-i18n-html]").forEach((node) => { node.innerHTML = t(node.dataset.i18nHtml); });
  root.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  root.querySelectorAll("[data-i18n-aria]").forEach((node) => { node.setAttribute("aria-label", t(node.dataset.i18nAria)); });
  root.querySelectorAll("[data-i18n-title]").forEach((node) => { node.title = t(node.dataset.i18nTitle); });
  document.documentElement.lang = current;
}

export function setLocale(code, { persist = true } = {}) {
  current = LOCALES[code] ? code : "en";
  if (persist) {
    try { localStorage.setItem(STORAGE_KEY, current); } catch { /* private mode */ }
  }
  applyStatic();
}
