document.addEventListener("DOMContentLoaded", function () {
    const RECS = window.RECOMMENDATION_DATA || {};
    const SENTIMENT = window.SENTIMENT_DATA || {};

    /* ------------------------------------------------------------
       1. Persona switcher — rewrites the hero banner instantly.
       ------------------------------------------------------------ */
    const pills = document.querySelectorAll(".mm-persona-pill");
    const hero = document.getElementById("storefront-hero");
    const heroGreeting = document.getElementById("hero-greeting");
    const heroMessage = document.getElementById("hero-message");
    const heroOffer = document.getElementById("hero-offer");
    const heroEmoji = document.getElementById("hero-emoji");

    function applyPersona(pill) {
        pills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");

        heroGreeting.textContent = pill.dataset.greeting;
        heroMessage.textContent = pill.dataset.message;
        heroOffer.textContent = pill.dataset.offer;
        heroEmoji.textContent = pill.dataset.emoji;
        hero.style.background = pill.dataset.theme;
    }

    pills.forEach((pill) => pill.addEventListener("click", () => applyPersona(pill)));
    if (pills.length) applyPersona(pills[0]);

    /* ------------------------------------------------------------
       2. Product detail panel — "Frequently bought together" +
          review snippets, matched against cached recommendation /
          sentiment data for this dataset.
       ------------------------------------------------------------ */
    const detailPanel = document.getElementById("product-detail-panel");
    const closeDetailBtn = document.getElementById("btn-close-product-detail");

    function findProductSentimentRow(productName) {
        const rows = SENTIMENT.product_breakdown || [];
        return rows.find((r) => r.product_name === productName);
    }

    function findFrequentlyBoughtWith(productName) {
        const rules = RECS.association_rules || [];
        const matches = [];
        rules.forEach((rule) => {
            const ifBought = String(rule.if_bought || "");
            const recommended = String(rule.recommended || "");
            if (ifBought.includes(productName)) {
                matches.push({ item: recommended, confidence: rule.confidence, lift: rule.lift });
            } else if (recommended.includes(productName)) {
                matches.push({ item: ifBought, confidence: rule.confidence, lift: rule.lift });
            }
        });
        if (matches.length) return matches.slice(0, 6);

        // Fallback: just show other popular items so the panel is never empty
        const popular = (RECS.popular_items || []).filter((p) => p.item !== productName);
        return popular.slice(0, 6).map((p) => ({ item: p.item, confidence: null, lift: null }));
    }

    function renderProductDetail(productName) {
        document.getElementById("detail-thumb").textContent = productName.charAt(0).toUpperCase();
        document.getElementById("detail-name").textContent = productName;

        const sentimentRow = findProductSentimentRow(productName);
        const sentimentLine = document.getElementById("detail-sentiment-line");
        if (sentimentRow) {
            sentimentLine.textContent =
                `${sentimentRow.pos_pct}% positive · ${sentimentRow.neu_pct}% neutral · ${sentimentRow.neg_pct}% negative (${sentimentRow.total} reviews)`;
        } else {
            sentimentLine.textContent = "No reviews analyzed for this product yet.";
        }

        // Frequently bought together / related items
        const recsBox = document.getElementById("detail-recs");
        const related = findFrequentlyBoughtWith(productName);
        if (related.length) {
            recsBox.innerHTML = related.map((r) => {
                const tag = r.confidence != null ? ` <small>(${r.confidence}% confidence)</small>` : "";
                return `<span class="mm-basket-chip">🧺 ${r.item}${tag}</span>`;
            }).join("");
        } else {
            recsBox.innerHTML = '<p class="text-muted small mb-0">Not enough data yet to suggest related items.</p>';
        }

        // Review snippets
        const reviewsBox = document.getElementById("detail-reviews");
        if (sentimentRow && SENTIMENT.comments_map && SENTIMENT.comments_map[sentimentRow.prod_id]) {
            const comments = SENTIMENT.comments_map[sentimentRow.prod_id];
            let html = "";
            (comments.positive || []).slice(0, 2).forEach((c) => {
                html += `<div class="mm-review-snippet positive">😊 ${escapeHtml(c)}</div>`;
            });
            (comments.negative || []).slice(0, 2).forEach((c) => {
                html += `<div class="mm-review-snippet negative">😕 ${escapeHtml(c)}</div>`;
            });
            reviewsBox.innerHTML = html || '<p class="text-muted small mb-0">No written comments to show.</p>';
        } else {
            reviewsBox.innerHTML = '<p class="text-muted small mb-0">Run Sentiment Analysis on this dataset to see real reviews here.</p>';
        }

        detailPanel.classList.remove("d-none");
        document.body.classList.add("mm-modal-open");
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    document.querySelectorAll(".mm-view-product").forEach((btn) => {
        btn.addEventListener("click", () => renderProductDetail(btn.dataset.product));
    });

    if (closeDetailBtn) {
        closeDetailBtn.addEventListener("click", () => {
            detailPanel.classList.add("d-none");
            document.body.classList.remove("mm-modal-open");
        });
    }

    /* ------------------------------------------------------------
       3. Personalized recommendations for a sample customer.
       ------------------------------------------------------------ */
    const customerSelect = document.getElementById("sample-customer-select");
    const customerRecsBox = document.getElementById("sample-customer-recs");

    function renderCustomerRecs() {
        if (!customerSelect || !customerRecsBox) return;
        const userId = customerSelect.value;
        const recs = (RECS.user_recommendations || {})[userId] || [];
        customerRecsBox.innerHTML = recs.length
            ? recs.map((item) => `<span class="mm-basket-chip">🎁 ${item}</span>`).join("")
            : '<span class="text-muted small">No personalized picks available for this customer.</span>';
    }

    if (customerSelect) {
        customerSelect.addEventListener("change", renderCustomerRecs);
        renderCustomerRecs();
    }
});
