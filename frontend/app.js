const state = { view: "overview", clientId: "", clients: [], data: {}, chat: [], assistantMode: "Grounded local answers · Hugging Face optional" };
const content = document.querySelector("#content");
const clientSelect = document.querySelector("#client-select");
const toast = document.querySelector("#toast");
const titles = { overview: "Overview", assistant: "AI assistant", transactions: "Transactions", portfolio: "Portfolio", risk: "Risk & profile", market: "Market & news" };

const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const number = (value, digits = 2) => Number(value || 0).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const compact = (value) => Number(value || 0).toLocaleString("en-US", { notation: "compact", maximumFractionDigits: 1 });
const money = (value, currency = "USD") => new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: 2 }).format(Number(value || 0));
const date = (value) => value ? new Date(`${value}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—";

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { headers: { "content-type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function notify(message) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2800);
}

function pageHeading(eyebrow, title, description, right = "") {
  return `<div class="page-heading"><div><div class="eyebrow">${escapeHtml(eyebrow)}</div><h1>${escapeHtml(title)}</h1><p>${escapeHtml(description)}</p></div>${right ? `<div class="date-label">${right}</div>` : ""}</div>`;
}

function sectionHead(title, action = "") {
  return `<div class="section-head"><h2>${escapeHtml(title)}</h2>${action}</div>`;
}

function stat(label, value, foot, icon, tone = "") {
  return `<article class="card stat-card"><div class="stat-top"><span>${escapeHtml(label)}</span><span class="stat-icon">${icon}</span></div><div class="stat-value ${tone}">${value}</div><div class="stat-foot">${escapeHtml(foot)}</div></article>`;
}

function table(headers, rows) {
  if (!rows.length) return `<div class="empty-state">No rows to show for this selection.</div>`;
  return `<div class="table-wrap"><table class="data-table"><thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
}

function transactionRows(rows) {
  return rows.map((row) => {
    const probability = row.fraud_probability;
    const risk = probability == null ? `<span class="muted">—</span>` : `<span class="pill ${probability >= 0.65 ? "red" : probability >= 0.35 ? "amber" : "green"}">${(probability * 100).toFixed(1)}% · ${row.predicted_fraud ? "Review" : "Low"}</span>`;
    return `<tr><td class="mono">${escapeHtml(row.transaction_id)}</td><td>${date(row.transaction_date)}</td><td>${escapeHtml(row.merchant)}</td><td>${escapeHtml(row.category)}</td><td>${escapeHtml(row.transaction_type)}</td><td class="mono">${number(row.amount)}</td><td>${risk}</td></tr>`;
  });
}

function overviewTransactionRows(rows) {
  return rows.map((row) => `<tr><td class="mono">${escapeHtml(row.transaction_id)}</td><td>${date(row.transaction_date)}</td><td>${escapeHtml(row.merchant)}</td><td>${escapeHtml(row.category)}</td><td class="mono">${number(row.amount)}</td><td>${row.fraud_probability == null ? "—" : `<span class="pill ${row.predicted_fraud ? "amber" : "green"}">${(row.fraud_probability * 100).toFixed(1)}%</span>`}</td></tr>`);
}

function renderOverview() {
  const summary = state.data.summary;
  const portfolio = state.data.portfolio;
  const risk = state.data.risk;
  const rows = state.data.transactions?.transactions || [];
  if (!summary) return `<div class="error-state">Client summary is not available. Generate the local demo data and try again.</div>`;
  const categoryEntries = Object.entries(summary.categories || {}).sort((a, b) => b[1] - a[1]).slice(0, 7);
  const max = Math.max(1, ...categoryEntries.map(([, count]) => count));
  const categoryHtml = categoryEntries.length ? categoryEntries.map(([name, count]) => `<div class="category-row"><span class="label">${escapeHtml(name)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, count / max * 100)}%"></div></div><span class="amount">${number(count, 0)}</span></div>`).join("") : `<div class="empty-state">No category activity found.</div>`;
  return `${pageHeading("CLIENT SNAPSHOT", "Good to see you", "A grounded view of synthetic activity, model scores, and portfolio data.", `Client · ${escapeHtml(state.clientId)}`)}
    <div class="grid stats-grid">
      ${stat("Transactions", compact(summary.transaction_count), "Generated activity records", "⇄")}
      ${stat("Total activity", number(summary.total_transaction_amount), "Across synthetic transactions", "↗")}
      ${stat("Model review flags", number(risk?.model_flagged_count, 0), `Of ${number(risk?.transaction_count_scored, 0)} transactions scored`, "⌁", risk?.model_flagged_count ? "negative" : "positive")}
      ${stat("Portfolio value", money(portfolio?.total_value_usd), `${portfolio?.position_count || 0} open positions`, "▥")}
    </div>
    <div class="grid two-col">
      <section class="card">${sectionHead("Recent transactions", `<button class="text-button" data-go="transactions">All transactions →</button>`)}
        ${table(["Transaction", "Date", "Merchant", "Category", "Amount", "Model score"], overviewTransactionRows(rows.slice(0, 6)))}
        <p class="table-note">Model scores use transaction features only. Synthetic labels are hidden from inference.</p>
      </section>
      <section class="card">${sectionHead("Activity by category", "<span>Transaction count</span>")}<div class="category-list">${categoryHtml}</div></section>
    </div>
    <div class="grid two-col spacer-top">
      <section class="card">${sectionHead("Portfolio snapshot", `<button class="text-button" data-go="portfolio">Open portfolio →</button>`)}
        <div class="metric-row"><div class="metric-mini"><span>Market value</span><strong>${money(portfolio?.total_value_usd)}</strong></div><div class="metric-mini"><span>Cost basis</span><strong>${money(portfolio?.total_cost_usd)}</strong></div><div class="metric-mini"><span>Unrealized change</span><strong class="${(portfolio?.gain_loss_usd || 0) >= 0 ? "positive" : "negative"}">${money(portfolio?.gain_loss_usd)}</strong></div></div>
        <p class="table-note">Generated holdings and historical prices; not live market data.</p>
      </section>
      <section class="card">${sectionHead("Client profile", `<button class="text-button" data-go="risk">View details →</button>`)}
        <div class="profile-banner"><div class="profile-avatar">${escapeHtml((summary.account?.name || state.clientId).split(/\s+/).map((part) => part[0]).join("").slice(0, 2))}</div><div><h2>${escapeHtml(summary.account?.name || state.clientId)}</h2><p>Credit score ${summary.credit_score} · ${escapeHtml(summary.risk_profile)} profile · KYC ${escapeHtml(summary.account?.kyc_status || "not available")}</p></div></div>
        <div class="callout">Generated data only. A model review score is not a finding of fraud or creditworthiness.</div>
      </section>
    </div>`;
}

function renderTransactions() {
  const rows = state.data.allTransactions?.transactions || [];
  const categories = [...new Set((state.data.allTransactions?.transactions || []).map((row) => row.category))].sort();
  return `${pageHeading("ACTIVITY", "Transactions", "Inspect the selected client’s generated transactions and saved-model risk scores.")}
    <div class="callout"><strong>Risk score is a model estimate.</strong> The classifier receives amount, time, category, type, and merchant. It does not read the generated fraud label.</div>
    <section class="card"><div class="toolbar"><select class="control" id="category-filter"><option value="">All categories</option>${categories.map((item) => `<option>${escapeHtml(item)}</option>`).join("")}</select><span class="table-note">Showing up to 100 newest rows</span></div>
      <div id="transaction-table">${table(["Transaction", "Date", "Merchant", "Category", "Type", "Amount", "Model risk"], transactionRows(rows))}</div>
      <p class="table-note">The returned API rows omit the synthetic label column.</p>
    </section>`;
}

function renderPortfolio() {
  const data = state.data.portfolio;
  if (!data) return `<div class="error-state">Portfolio data could not be loaded. Run the full synthetic data generator.</div>`;
  const rows = data.positions.map((row) => `<tr><td><strong>${escapeHtml(row.symbol)}</strong><div class="table-note">${escapeHtml(row.sector)}</div></td><td>${escapeHtml(row.industry)}</td><td class="mono">${number(row.quantity, 4)}</td><td class="mono">${money(row.avg_cost_usd)}</td><td class="mono">${money(row.market_value_usd)}</td><td class="mono">${money(row.gain_loss_usd)}</td><td><span class="pill ${row.gain_loss_pct >= 0 ? "green" : "red"}">${number(row.gain_loss_pct)}%</span></td></tr>`);
  return `${pageHeading("BROKERAGE BOOK", "Portfolio", "A deterministic snapshot computed from the generated ledger and market prices.", `Price date · ${date(data.as_of)}`)}
    <div class="grid stats-grid">${stat("Market value", money(data.total_value_usd), "Generated closing prices", "▥")}${stat("Cost basis", money(data.total_cost_usd), "Average cost × current units", "⌗")}${stat("Unrealized change", money(data.gain_loss_usd), `${number(data.gain_loss_pct)}% on cost basis`, "↗", data.gain_loss_usd >= 0 ? "positive" : "negative")}${stat("Open positions", number(data.position_count, 0), "Distinct instruments held", "◈")}</div>
    <section class="card">${sectionHead("Holdings", "<span>All values in USD</span>")}${table(["Instrument", "Industry", "Quantity", "Avg. cost", "Market value", "Change", "Return"], rows)}<p class="table-note">Unrealized change is a snapshot calculation, not realized profit or a recommendation.</p></section>`;
}

function renderRisk() {
  const profile = state.data.profile;
  const risk = state.data.risk;
  if (!profile || !risk) return `<div class="error-state">Risk or profile data is unavailable. Confirm that the model and generated datasets exist.</div>`;
  const flagRate = risk.transaction_count_scored ? risk.model_flagged_count / risk.transaction_count_scored * 100 : 0;
  const account = profile.account || {};
  return `${pageHeading("PROFILE & RISK", "Risk and client profile", "Synthetic profile fields and local model output, kept separate from real financial decisions.")}
    <div class="grid profile-layout">
      <section class="card"><div class="profile-banner"><div class="profile-avatar">${escapeHtml((account.name || state.clientId).split(/\s+/).map((part) => part[0]).join("").slice(0, 2))}</div><div><h2>${escapeHtml(account.name || state.clientId)}</h2><p>${escapeHtml(state.clientId)} · KYC ${escapeHtml(account.kyc_status || "not available")}</p></div></div>
        <div class="grid profile-grid"><div class="card detail-card"><div class="detail-label">Age</div><div class="detail-value">${profile.age}</div></div><div class="card detail-card"><div class="detail-label">Credit score</div><div class="detail-value">${profile.credit_score}</div></div><div class="card detail-card"><div class="detail-label">Generated risk group</div><div class="detail-value">${escapeHtml(profile.risk_profile)}</div></div><div class="card detail-card"><div class="detail-label">Annual income</div><div class="detail-value">${number(profile.annual_income, 0)}</div></div><div class="card detail-card"><div class="detail-label">Income band</div><div class="detail-value">${escapeHtml(account.annual_income_band || "—")}</div></div><div class="card detail-card"><div class="detail-label">Account opened</div><div class="detail-value">${date(account.account_opened)}</div></div></div>
        <p class="table-note">Synthetic client profile and brokerage KYC fields are separately generated; risk categories may differ.</p>
      </section>
      <section class="card">${sectionHead("Saved fraud classifier", "<span>Scikit-learn Random Forest</span>")}
        <div class="metric-row"><div class="metric-mini"><span>Scored transactions</span><strong>${number(risk.transaction_count_scored, 0)}</strong></div><div class="metric-mini"><span>Model review flags</span><strong>${number(risk.model_flagged_count, 0)}</strong></div><div class="metric-mini"><span>Flagged share</span><strong>${number(flagRate, 1)}%</strong></div></div>
        <div class="detail-label">Selected synthetic transactions above threshold</div><div class="risk-bar ${flagRate > 10 ? "high" : flagRate > 4 ? "" : "low"}"><i style="width:${Math.min(flagRate, 100)}%"></i></div>
        <div class="table-wrap spacer-top">${table(["Transaction", "Date", "Amount", "Probability"], risk.highest_risk_transactions.map((row) => `<tr><td class="mono">${escapeHtml(row.transaction_id)}</td><td>${date(row.transaction_date)}</td><td class="mono">${number(row.amount)}</td><td><span class="pill ${row.predicted_fraud ? "amber" : "green"}">${(row.fraud_probability * 100).toFixed(1)}%</span></td></tr>`))}</div>
        ${renderShap(state.data.riskExplanation)}
        <div class="callout spacer-top">Synthetic training labels only. A flagged transaction does not establish actual fraud. Credit scores and synthetic risk groups are demonstration data.</div>
      </section>
    </div>`;
}

function renderShap(data) {
  if (!data) return `<div class="callout spacer-top">Feature explanations are not available. Install project requirements to enable SHAP.</div>`;
  if (!data.contributions?.length) return `<p class="table-note spacer-top">${escapeHtml(data.note || "No feature explanation available.")}</p>`;
  const largest = Math.max(0.00001, ...data.contributions.map((item) => Math.abs(item.impact)));
  const bars = data.contributions.map((item) => `<div class="category-row"><span class="label">${escapeHtml(item.feature)}</span><div class="bar-track"><div class="bar-fill ${item.impact < 0 ? "shap-negative" : "shap-positive"}" style="width:${Math.max(2, Math.abs(item.impact) / largest * 100)}%"></div></div><span class="amount ${item.impact >= 0 ? "negative" : "positive"}">${item.impact >= 0 ? "+" : ""}${number(item.impact, 3)}</span></div>`).join("");
  return `<section class="spacer-top">${sectionHead("Why this model scored the top transaction", `<span>${escapeHtml(data.transaction_id || "Selected record")}</span>`)}<div class="category-list">${bars}</div><p class="table-note">Positive impact raises the model output; negative impact lowers it. ${escapeHtml(data.note || "")}</p></section>`;
}

function renderMarket() {
  const data = state.data.market || { quotes: [], news: [] };
  const q = new URLSearchParams(location.hash.split("?")[1] || "").get("symbol") || "";
  const cards = data.quotes.map((quote) => `<article class="card quote-card" data-symbol="${escapeHtml(quote.symbol)}"><div class="section-head"><span class="quote-symbol">${escapeHtml(quote.symbol)}</span><span class="pill">${escapeHtml(quote.exchange)}</span></div><div class="quote-price">${money(quote.close)}</div><div class="quote-meta">${escapeHtml(quote.sector)} · ${date(quote.date)}</div></article>`).join("");
  const news = data.news.map((item) => `<article class="news-item"><div class="news-date">${date(item.date)}<br><span class="pill">${escapeHtml(item.symbol)}</span></div><div><strong>${escapeHtml(item.headline)}</strong><p>${escapeHtml(item.body)}</p></div></article>`).join("");
  return `${pageHeading("GENERATED DATA", "Market & news", "Browse the locally generated price history and synthetic newswire entries.")}
    <div class="toolbar"><input class="search-input" id="market-search" value="${escapeHtml(q)}" placeholder="Filter by symbol, e.g. NVDA" maxlength="5"><button class="primary-button" id="market-search-button">Search symbol</button><button class="text-button" id="market-reset">Show all</button></div>
    <div class="grid quote-grid">${cards || `<div class="empty-state">No generated quote matches that symbol.</div>`}</div>
    <section class="card spacer-top">${sectionHead("Synthetic newswire", `<span>Generated headlines, not live reporting</span>`)}<div class="news-list">${news || `<div class="empty-state">No generated news matches this filter.</div>`}</div></section>`;
}

function renderAssistant() {
  const items = state.chat.map((item) => `<article class="message ${item.role}">${escapeHtml(item.text)}${item.sources?.length ? `<span class="message-meta">${escapeHtml(item.agent)} · ${escapeHtml(item.sources.join(" · "))}${item.llm_used ? " · HF explanation verified" : " · deterministic result"}</span>` : ""}</article>`).join("");
  return `${pageHeading("GROUNDED Q&A", "AI assistant", "Ask about the selected client. Python retrieves and calculates; Hugging Face can explain when configured.")}
    <section class="card chat-card"><div class="chat-header"><h2>FinGuard assistant</h2><span id="assistant-mode">${escapeHtml(state.assistantMode)}</span></div>
      <div class="chat-messages" id="chat-messages">${items || `<article class="message assistant">Ask about transactions, portfolio value, client profile/KYC, credit and model risk, or generated market prices and news.<span class="message-meta">Evidence comes from local synthetic datasets</span></article>`}</div>
      <div class="suggestions"><button class="suggestion" data-question="Summarize this client's recent transactions">Summarize transactions</button><button class="suggestion" data-question="Show this client's portfolio and unrealized return">Portfolio snapshot</button><button class="suggestion" data-question="What are the client's credit score and model risk flags?">Risk and score</button><button class="suggestion" data-question="What is the latest generated market news?">Market news</button><button class="suggestion" data-question="Show the selected client's profile and KYC status">Profile and KYC</button></div>
      <form class="chat-form" id="chat-form"><input class="chat-input" id="chat-input" maxlength="500" placeholder="Ask a question about this synthetic client…" autocomplete="off" required><button class="primary-button" id="send-button">Ask assistant ↑</button></form>
    </section><p class="table-note">The assistant is scoped to the selected client, declines investment advice, and checks generated answers against Python evidence. Hugging Face is optional; a grounded local response always works without a token.</p>`;
}

async function loadViewData(view = state.view) {
  const id = encodeURIComponent(state.clientId);
  const fetchSafely = async (key, path) => {
    try { state.data[key] = await api(path); } catch (error) { state.data[key] = null; console.warn(`${key}: ${error.message}`); }
  };
  if (view === "overview") {
    await Promise.all([fetchSafely("summary", `/clients/${id}/summary`), fetchSafely("portfolio", `/clients/${id}/portfolio`), fetchSafely("risk", `/clients/${id}/risk`), fetchSafely("transactions", `/clients/${id}/transactions?limit=10&predict=true`)]);
  } else if (view === "transactions") {
    await fetchSafely("allTransactions", `/clients/${id}/transactions?limit=100&predict=true`);
  } else if (view === "portfolio") {
    await fetchSafely("portfolio", `/clients/${id}/portfolio`);
  } else if (view === "risk") {
    await Promise.all([fetchSafely("profile", `/clients/${id}/profile`), fetchSafely("risk", `/clients/${id}/risk`), fetchSafely("riskExplanation", `/clients/${id}/risk/explanation`)]);
  } else if (view === "market") {
    const symbol = new URLSearchParams(location.hash.split("?")[1] || "").get("symbol");
    await fetchSafely("market", `/market${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ""}`);
  }
}

function render() {
  document.querySelector("#page-title").textContent = titles[state.view];
  document.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === state.view));
  const renderer = { overview: renderOverview, assistant: renderAssistant, transactions: renderTransactions, portfolio: renderPortfolio, risk: renderRisk, market: renderMarket }[state.view];
  content.innerHTML = renderer();
  content.querySelectorAll("[data-go]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.go)));
  if (state.view === "transactions") bindTransactionFilter();
  if (state.view === "assistant") bindAssistant();
  if (state.view === "market") bindMarket();
}

async function navigate(view) {
  state.view = view;
  history.replaceState(null, "", `#${view}`);
  render();
  if (view === "assistant") {
    try {
      const historyData = await api(`/assistant/history/${encodeURIComponent(state.clientId)}`);
      state.chat = (historyData.messages || []).flatMap((item) => [{ role: "user", text: item.question }, { role: "assistant", text: item.answer }]);
      if (state.view === view) render();
    } catch (error) { notify(error.message); }
  }
  if (view !== "assistant") {
    await loadViewData(view);
    if (state.view === view) render();
  }
}

function bindTransactionFilter() {
  document.querySelector("#category-filter")?.addEventListener("change", async (event) => {
    const search = new URLSearchParams({ limit: "100", predict: "true" });
    if (event.target.value) search.set("category", event.target.value);
    try {
      const data = await api(`/clients/${encodeURIComponent(state.clientId)}/transactions?${search}`);
      document.querySelector("#transaction-table").innerHTML = table(["Transaction", "Date", "Merchant", "Category", "Type", "Amount", "Model risk"], transactionRows(data.transactions));
    } catch (error) { notify(error.message); }
  });
}

function bindMarket() {
  const search = async () => {
    const value = document.querySelector("#market-search").value.trim().toUpperCase();
    location.hash = value ? `market?symbol=${encodeURIComponent(value)}` : "market";
    await loadViewData("market"); render();
  };
  document.querySelector("#market-search-button")?.addEventListener("click", search);
  document.querySelector("#market-search")?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); search(); } });
  document.querySelector("#market-reset")?.addEventListener("click", async () => { location.hash = "market"; await loadViewData("market"); render(); });
  content.querySelectorAll("[data-symbol]").forEach((card) => card.addEventListener("click", () => { document.querySelector("#market-search").value = card.dataset.symbol; search(); }));
}

function bindAssistant() {
  const messages = document.querySelector("#chat-messages");
  messages.scrollTop = messages.scrollHeight;
  document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => submitQuestion(button.dataset.question)));
  document.querySelector("#chat-form")?.addEventListener("submit", (event) => { event.preventDefault(); submitQuestion(document.querySelector("#chat-input").value); });
}

async function submitQuestion(question) {
  question = question.trim();
  if (!question) return;
  state.chat.push({ role: "user", text: question });
  state.chat.push({ role: "assistant", text: "Checking local evidence…" });
  render();
  const send = document.querySelector("#send-button");
  if (send) send.disabled = true;
  try {
    const result = await api("/assistant", { method: "POST", body: JSON.stringify({ client_id: state.clientId, question }) });
    state.chat[state.chat.length - 1] = { role: "assistant", text: result.answer, agent: result.agent, sources: result.sources, llm_used: result.llm_used };
    if (result.refused) notify("The assistant kept this request within the local demo boundaries.");
  } catch (error) {
    state.chat[state.chat.length - 1] = { role: "assistant", text: `I couldn't read the local data: ${error.message}` };
  }
  render();
  document.querySelector("#chat-input")?.focus();
}

async function boot() {
  try {
    const auth = await api("/auth/me");
    if (!auth.authenticated) {
      document.querySelector(".app-shell").hidden = true;
      const loginScreen = document.querySelector("#login-screen");
      loginScreen.hidden = false;
      document.querySelector("#login-form").addEventListener("submit", async (event) => {
        event.preventDefault();
        const form = new FormData(event.currentTarget);
        const errorBox = document.querySelector("#login-error");
        errorBox.textContent = "";
        try {
          await api("/auth/login", { method: "POST", body: JSON.stringify({ username: form.get("username"), password: form.get("password") }) });
          location.reload();
        } catch (error) { errorBox.textContent = error.message; }
      });
      return;
    }
    document.querySelector("#logout-button").addEventListener("click", async () => {
      await api("/auth/logout", { method: "POST" });
      location.reload();
    });
    const [config, clientResult] = await Promise.all([api("/config"), api("/clients")]);
    state.clients = clientResult.clients;
    const label = config.llm_enabled ? `Hugging Face · ${config.llm_model}` : "Grounded local mode · HF token not set";
    document.querySelector("#model-status").innerHTML = `<span class="status-dot"></span>${escapeHtml(label)}`;
    state.assistantMode = config.llm_enabled ? "Agno + Hugging Face enabled" : "Grounded local answers · Hugging Face optional";
    clientSelect.innerHTML = state.clients.map((client) => `<option value="${escapeHtml(client.client_id)}">${escapeHtml(client.name)} · ${escapeHtml(client.client_id)}</option>`).join("");
    state.clientId = state.clients[0]?.client_id || "";
    clientSelect.value = state.clientId;
    document.querySelector("#client-avatar").textContent = state.clientId.slice(-2) || "CL";
    if (!state.clientId) throw new Error("The data generator returned no clients.");
    const requestedView = location.hash.slice(1).split("?")[0];
    if (titles[requestedView]) state.view = requestedView;
    clientSelect.addEventListener("change", async () => {
      state.clientId = clientSelect.value;
      document.querySelector("#client-avatar").textContent = state.clientId.slice(-2);
      state.data = {};
      state.chat = [];
      await loadViewData(); render();
    });
    document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
    render();
    if (state.view !== "assistant") { await loadViewData(); render(); }
    else {
      const historyData = await api(`/assistant/history/${encodeURIComponent(state.clientId)}`);
      state.chat = (historyData.messages || []).flatMap((item) => [{ role: "user", text: item.question }, { role: "assistant", text: item.answer }]);
      render();
    }
  } catch (error) {
    content.innerHTML = `<div class="error-state"><strong>FinGuard could not load its local data.</strong><br>${escapeHtml(error.message)}<br><br>Run <code>python -m src.data.generate_data</code> and <code>python -m src.ml.train_model</code>, then refresh.</div>`;
  }
}

boot();
