/* =========================================================================
   MarketMind AI — "Fun layer"
   Purely additive: mascot tips, confetti celebrations, an animated step
   tracker, count-up numbers, and small achievement toasts.
   Nothing here overrides or removes any existing AJAX/business logic —
   it only *observes* the DOM the app already updates and reacts to it.
   ========================================================================= */

(function () {
    "use strict";

    /* ----------------------------------------------------------------
       1. Mascot: contextual, plain-language tips per section
       ---------------------------------------------------------------- */
    const MASCOT_TIPS = {
        overview: [
            "Hi, I'm Miko 🤖 your data buddy! This page peeks at your file and checks which cool tricks we can run on it.",
            "See those 'Suitable' badges? That's me checking if your columns have what each tool needs — no guesswork required!"
        ],
        preprocessing: [
            "Think of this like tidying your room before guests arrive 🧹 — filling gaps, fixing types, and cleaning messy text.",
            "Missing values are just empty boxes in your data. Here you get to decide how to fill them in!"
        ],
        clustering: [
            "RFM is just 3 friendly questions about each customer: How Recently did they buy? How Frequently? How much Money did they spend? 💰",
            "Once we answer those 3 questions, we can sort customers into groups — like 'Champions' or 'At Risk' — automatically!"
        ],
        recommendation: [
            "This is the same idea behind 'You might also like...' on Netflix or Amazon 🎬 — we spot patterns in what people buy together.",
            "No magic here — just counting which items show up together a lot, then suggesting the rest!"
        ],
        reviews: [
            "This is a mood detector for text 😊😐☹️ — it reads each review and decides if it sounds happy, neutral, or unhappy.",
            "Try it on real reviews — you'll see the donut chart fill up with green (happy), grey (neutral) and red (unhappy)!"
        ]
    };

    let mascotEl, bubbleEl, bubbleTextEl, avatarEl;
    let tipRotation = {};

    function buildMascot() {
        if (document.getElementById("mm-mascot-wrap")) return;

        mascotEl = document.createElement("div");
        mascotEl.id = "mm-mascot-wrap";
        mascotEl.innerHTML = `
            <div id="mm-mascot-bubble" role="status">
                <span class="mm-bubble-close" title="Hide">&times;</span>
                <span id="mm-mascot-text"></span>
            </div>
            <button id="mm-mascot-avatar" type="button" title="Need a hint?">🤖</button>
        `;
        document.body.appendChild(mascotEl);

        bubbleEl = document.getElementById("mm-mascot-bubble");
        bubbleTextEl = document.getElementById("mm-mascot-text");
        avatarEl = document.getElementById("mm-mascot-avatar");

        bubbleEl.querySelector(".mm-bubble-close").addEventListener("click", () => {
            bubbleEl.style.display = "none";
            avatarEl.classList.add("mm-hidden-bubble");
        });

        avatarEl.addEventListener("click", () => {
            bubbleEl.style.display = "block";
            avatarEl.classList.remove("mm-hidden-bubble");
            cycleTip(currentSection());
        });
    }

    function currentSection() {
        const active = document.querySelector(".sidebar .nav-link.active");
        return active ? active.dataset.section : "overview";
    }

    function say(text) {
        if (!bubbleEl) return;
        bubbleEl.style.display = "block";
        avatarEl.classList.remove("mm-hidden-bubble");
        bubbleEl.classList.remove("mm-pop");
        // restart animation
        void bubbleEl.offsetWidth;
        bubbleEl.classList.add("mm-pop");
        bubbleTextEl.textContent = text;
    }

    function cycleTip(sectionId) {
        const tips = MASCOT_TIPS[sectionId] || MASCOT_TIPS.overview;
        const idx = tipRotation[sectionId] || 0;
        say(tips[idx % tips.length]);
        tipRotation[sectionId] = idx + 1;
    }

    /* ----------------------------------------------------------------
       2. Confetti celebrations (via canvas-confetti CDN, loaded in
          dashboard.html). Falls back to a no-op if the library failed
          to load, so nothing ever throws.
       ---------------------------------------------------------------- */
    function celebrate(strength) {
        if (typeof confetti !== "function") return;
        const particleCount = strength === "big" ? 140 : 70;
        confetti({
            particleCount,
            spread: 75,
            startVelocity: 45,
            origin: { y: 0.6 },
            colors: ["#7c3aed", "#ec4899", "#3b82f6", "#14b8a6", "#f59e0b"]
        });
    }

    /* ----------------------------------------------------------------
       3. Achievement toasts
       ---------------------------------------------------------------- */
    function ensureToastStack() {
        let stack = document.getElementById("mm-toast-stack");
        if (!stack) {
            stack = document.createElement("div");
            stack.id = "mm-toast-stack";
            document.body.appendChild(stack);
        }
        return stack;
    }

    function achievement(title, subtitle) {
        const stack = ensureToastStack();
        const toast = document.createElement("div");
        toast.className = "mm-toast";
        toast.innerHTML = `${title}${subtitle ? `<small>${subtitle}</small>` : ""}`;
        stack.appendChild(toast);
        setTimeout(() => {
            toast.style.transition = "opacity 0.4s ease, transform 0.4s ease";
            toast.style.opacity = "0";
            toast.style.transform = "translateX(20px)";
            setTimeout(() => toast.remove(), 400);
        }, 3600);
    }

    /* ----------------------------------------------------------------
       4. Step tracker — lights up sidebar items once their result
          panel has appeared at least once.
       ---------------------------------------------------------------- */
    const STEP_MAP = {
        "segmentation-summary": { section: "clustering", label: "Segmentation", msg: "You just grouped your customers into cohorts!" },
        "recommendation-results": { section: "recommendation", label: "Recommendations", msg: "Fresh product recommendations, served!" },
        "sentiment-results": { section: "reviews", label: "Reviews", msg: "Reviews analyzed — moods decoded!" }
    };

    function markStepDone(sectionId) {
        const link = document.querySelector(`.sidebar .nav-link[data-section="${sectionId}"]`);
        if (link && !link.classList.contains("mm-step-done")) {
            link.classList.add("mm-step-done");
        }
        updateStepper();
        setTimeout(maybeFireFinale, 900);
    }

    function watchResultPanels() {
        Object.keys(STEP_MAP).forEach((id) => {
            const el = document.getElementById(id);
            if (!el) return;
            const observer = new MutationObserver(() => {
                if (!el.classList.contains("d-none")) {
                    const info = STEP_MAP[id];
                    markStepDone(info.section);
                    celebrate(info.section === "clustering" ? "big" : "normal");
                    achievement("🎉 Nice work!", info.msg);
                    observer.disconnect();
                }
            });
            observer.observe(el, { attributes: true, attributeFilter: ["class"] });
        });
    }

    /* ----------------------------------------------------------------
       5. Progress stepper UI (built once, injected under the navbar)
       ---------------------------------------------------------------- */
    const STEPS = [
        { id: "overview", label: "Explore", icon: "1" },
        { id: "preprocessing", label: "Clean", icon: "2" },
        { id: "clustering", label: "Segment", icon: "3" },
        { id: "recommendation", label: "Recommend", icon: "4" },
        { id: "reviews", label: "Reviews", icon: "5" }
    ];

    function buildStepper() {
        const anchor = document.getElementById("mm-stepper-anchor");
        if (!anchor || document.getElementById("mm-stepper")) return;

        const stepper = document.createElement("div");
        stepper.id = "mm-stepper";
        stepper.className = "mm-stepper mb-4 mm-fade-up";

        STEPS.forEach((step, i) => {
            const stepEl = document.createElement("div");
            stepEl.className = "mm-step";
            stepEl.dataset.step = step.id;
            stepEl.innerHTML = `<span class="mm-step-dot">${step.icon}</span><span>${step.label}</span>`;
            stepper.appendChild(stepEl);
            if (i < STEPS.length - 1) {
                const sep = document.createElement("div");
                sep.className = "mm-step-sep";
                stepper.appendChild(sep);
            }
        });

        anchor.appendChild(stepper);
        updateStepper();
    }

    function updateStepper() {
        const stepper = document.getElementById("mm-stepper");
        if (!stepper) return;
        const active = currentSection();
        STEPS.forEach((step) => {
            const el = stepper.querySelector(`[data-step="${step.id}"]`);
            if (!el) return;
            const done = document.querySelector(`.sidebar .nav-link[data-section="${step.id}"]`)
                ?.classList.contains("mm-step-done");
            el.classList.toggle("mm-current", step.id === active);
            el.classList.toggle("mm-done", !!done && step.id !== active);
            if (done) el.querySelector(".mm-step-dot").textContent = "✓";
        });
    }

    /* ----------------------------------------------------------------
       6. Count-up animation for KPI numbers
       ---------------------------------------------------------------- */
    function animateCountUp(el) {
        const target = parseInt((el.textContent || "0").replace(/[^\d]/g, ""), 10) || 0;
        if (target === 0) return;
        const duration = 900;
        const start = performance.now();

        function tick(now) {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(target * eased).toLocaleString();
            if (progress < 1) requestAnimationFrame(tick);
            else el.textContent = target.toLocaleString();
        }
        requestAnimationFrame(tick);
    }

    function runCountUps(scope) {
        (scope || document).querySelectorAll(".mm-countup").forEach(animateCountUp);
    }

    /* ----------------------------------------------------------------
       7. Wire section switching (additive listener; doesn't touch the
          existing dashboard.js click handler that shows/hides sections)
       ---------------------------------------------------------------- */
    function wireSectionSwitching() {
        document.querySelectorAll(".sidebar .nav-link").forEach((btn) => {
            btn.addEventListener("click", function () {
                const sectionId = this.dataset.section;
                updateStepper();
                setTimeout(() => cycleTip(sectionId), 250);
                const target = document.getElementById(sectionId);
                if (target) runCountUps(target);
            });
        });
    }

    /* ----------------------------------------------------------------
       8. Ambient background blobs — purely decorative, sits behind
          everything at z-index -1.
       ---------------------------------------------------------------- */
    function injectBackgroundBlobs() {
        if (document.getElementById("mm-bg-blobs")) return;
        const wrap = document.createElement("div");
        wrap.id = "mm-bg-blobs";
        wrap.innerHTML = "<span></span><span></span><span></span>";
        document.body.appendChild(wrap);
    }

    /* ----------------------------------------------------------------
       9. Playful alert system — intercepts window.alert() so every
          success/error message in the app becomes a friendly animated
          toast instead of a blocking native popup. Purely additive:
          no other file needs to change what it calls.
       ---------------------------------------------------------------- */
    const ERROR_HINTS = ["error", "fail", "invalid", "unsuitable", "missing", "unable", "could not", "wrong"];

    function ensureAlertStack() {
        let stack = document.getElementById("mm-alert-stack");
        if (!stack) {
            stack = document.createElement("div");
            stack.id = "mm-alert-stack";
            document.body.appendChild(stack);
        }
        return stack;
    }

    function prettyAlert(message) {
        const text = String(message == null ? "" : message);
        const isError = ERROR_HINTS.some((w) => text.toLowerCase().includes(w));
        const stack = ensureAlertStack();

        const card = document.createElement("div");
        card.className = "mm-alert-card" + (isError ? " mm-alert-error" : "");
        card.innerHTML = `
            <span class="mm-alert-icon">${isError ? "😅" : "✨"}</span>
            <span>${text}</span>
            <span class="mm-alert-close" title="Dismiss">&times;</span>
        `;
        stack.appendChild(card);

        const dismiss = () => {
            if (!card.parentNode) return;
            card.classList.add("mm-alert-leaving");
            setTimeout(() => card.remove(), 260);
        };
        card.querySelector(".mm-alert-close").addEventListener("click", dismiss);
        setTimeout(dismiss, isError ? 5200 : 3400);

        if (!isError) celebrate("normal");
    }

    function overrideNativeAlert() {
        try {
            window.alert = prettyAlert;
        } catch (e) { /* no-op if locked down */ }
    }

    /* ----------------------------------------------------------------
       10. Explain-it tooltips — turns any .mm-explain badge into a
           Bootstrap tooltip so jargon gets a plain-language definition
           on hover/tap.
       ---------------------------------------------------------------- */
    function initExplainTooltips(scope) {
        if (typeof bootstrap === "undefined" || !bootstrap.Tooltip) return;
        (scope || document).querySelectorAll(".mm-explain").forEach((el) => {
            if (el.dataset.mmTooltipReady) return;
            el.dataset.mmTooltipReady = "1";
            new bootstrap.Tooltip(el, { trigger: "hover focus click" });
        });
    }

    /* ----------------------------------------------------------------
       11. Fun loading phrases — while any spinner-border is visible,
           rotate its label through playful, context-aware phrases
           instead of a static "Processing...".
       ---------------------------------------------------------------- */
    const LOADING_PHRASES = {
        clustering: ["Sorting customers into cohorts…", "Asking the RFM questions…", "Counting recency, frequency, money…", "Almost there…"],
        recommendation: ["Spotting patterns in purchases…", "Teaching the AI your catalog…", "Finding things bought together…", "Almost there…"],
        reviews: ["Reading through the reviews…", "Measuring the mood…", "Counting smiles and frowns…", "Almost there…"],
        preprocessing: ["Tidying up the data…", "Filling in the gaps…", "Almost there…"],
        default: ["Crunching the numbers…", "Working some AI magic…", "Almost there…"]
    };

    function watchSpinners() {
        const seen = new WeakSet();
        const observer = new MutationObserver(() => {
            document.querySelectorAll(".spinner-border").forEach((spinner) => {
                if (seen.has(spinner)) return;
                seen.add(spinner);

                const btn = spinner.closest("button");
                if (!btn) return;
                const section = btn.closest(".content-section");
                const phrases = LOADING_PHRASES[section ? section.id : "default"] || LOADING_PHRASES.default;

                let i = 0;
                const spinnerHTML = spinner.outerHTML;
                const interval = setInterval(() => {
                    if (!document.body.contains(spinner) || !btn.disabled) {
                        clearInterval(interval);
                        return;
                    }
                    i = (i + 1) % phrases.length;
                    btn.innerHTML = spinnerHTML + " " + phrases[i];
                }, 1100);
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    /* ----------------------------------------------------------------
       12. Grand finale — when all three analysis modules have been
           completed at least once, throw a bigger celebration.
       ---------------------------------------------------------------- */
    let finaleFired = false;
    const FINALE_STEPS = ["clustering", "recommendation", "reviews"];

    function maybeFireFinale() {
        if (finaleFired) return;
        const allDone = FINALE_STEPS.every((id) =>
            document.querySelector(`.sidebar .nav-link[data-section="${id}"]`)?.classList.contains("mm-step-done")
        );
        if (!allDone) return;
        finaleFired = true;

        // Two-sided confetti cannon for extra wow factor
        const burst = (originX) => {
            if (typeof confetti !== "function") return;
            confetti({
                particleCount: 100,
                angle: originX < 0.5 ? 60 : 120,
                spread: 65,
                startVelocity: 55,
                origin: { x: originX, y: 0.7 },
                colors: ["#7c3aed", "#ec4899", "#3b82f6", "#14b8a6", "#f59e0b"]
            });
        };
        burst(0.1);
        setTimeout(() => burst(0.9), 200);
        setTimeout(() => burst(0.5), 400);

        const overlay = document.createElement("div");
        overlay.id = "mm-finale-overlay";
        overlay.innerHTML = `
            <div class="mm-finale-card">
                <span class="mm-finale-emoji">🏆</span>
                <h3>You just did real data science!</h3>
                <p>You segmented customers, generated recommendations, and decoded review sentiment — all three, all by yourself. That's the whole analytics pipeline. 🎓</p>
                <button class="btn btn-primary fw-bold px-4" id="mm-finale-close">Keep exploring</button>
            </div>
        `;
        document.body.appendChild(overlay);
        const close = () => overlay.remove();
        overlay.querySelector("#mm-finale-close").addEventListener("click", close);
        overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    }

    /* ----------------------------------------------------------------
       13. Resume where you left off — when a dataset is reloaded from
           history, mark any already-completed steps and jump straight
           to the furthest section instead of always landing on Overview.
       ---------------------------------------------------------------- */
    const CACHE_TO_SECTION = { segmentation: "clustering", recommendation: "recommendation", reviews: "reviews" };

    function restoreProgressFromCache() {
        const cached = window.CACHED_RESULTS || {};
        let anyCached = false;

        Object.keys(CACHE_TO_SECTION).forEach((key) => {
            if (!cached[key]) return;
            anyCached = true;
            const link = document.querySelector(`.sidebar .nav-link[data-section="${CACHE_TO_SECTION[key]}"]`);
            if (link) link.classList.add("mm-step-done");
        });
        updateStepper();

        // Already fully done in a previous session — don't re-blast confetti
        // every time the dataset is reopened, just quietly consider it fired.
        if (Object.keys(CACHE_TO_SECTION).every((key) => cached[key])) {
            finaleFired = true;
        }

        const target = window.ACTIVE_SECTION;
        if (target && target !== "overview") {
            const link = document.querySelector(`.sidebar .nav-link[data-section="${target}"]`);
            if (link) {
                link.click();
                if (anyCached) {
                    setTimeout(() => achievement("👋 Welcome back!", "Picking up right where you left off."), 400);
                }
            }
        }
    }

    /* ----------------------------------------------------------------
       Boot
       ---------------------------------------------------------------- */
    document.addEventListener("DOMContentLoaded", function () {
        overrideNativeAlert();
        injectBackgroundBlobs();
        buildMascot();
        buildStepper();
        watchResultPanels();
        wireSectionSwitching();
        watchSpinners();
        initExplainTooltips();
        runCountUps(document);
        restoreProgressFromCache();

        // Warm welcome on first load of a dataset
        setTimeout(() => cycleTip("overview"), 600);

        // Little "welcome" sparkle so the first impression feels alive
        if (document.getElementById("overview")) {
            setTimeout(() => celebrate("normal"), 350);
        }
    });

    // Expose a couple of hooks in case other inline scripts want to use them
    window.MM_FUN = { celebrate, achievement, say: cycleTip, alert: prettyAlert };
})();
