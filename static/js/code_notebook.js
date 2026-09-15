/* =========================================================================
   MarketMind AI — Workspace mode toggle (AI Analyser <-> Code) and the
   Code Notebook cell engine (a small Colab/Jupyter-style client for the
   /code/init, /code/execute, /code/reset backend endpoints).
   ========================================================================= */

document.addEventListener("DOMContentLoaded", function () {

    // =====================================================================
    // 1. AI Analyser <-> Code mode toggle
    // =====================================================================
    const analyserBtn = document.getElementById("mode-btn-analyser");
    const codeBtn = document.getElementById("mode-btn-code");
    const analyserMode = document.getElementById("ai-analyser-mode");
    const codeMode = document.getElementById("code-mode");
    const analyserNav = document.getElementById("analyser-nav");
    const codeNav = document.getElementById("code-nav");
    const sectionTitle = document.getElementById("current-section-title");
    const sectionIcon = document.getElementById("current-section-icon");

    let notebookInitialized = false;
    let currentMode = "analyser";

    function activateAnalyserMode() {
        currentMode = "analyser";
        analyserBtn?.classList.add("active");
        codeBtn?.classList.remove("active");
        if (analyserMode) analyserMode.style.display = "block";
        if (codeMode) codeMode.style.display = "none";
        analyserNav?.classList.remove("d-none");
        codeNav?.classList.add("d-none");

        const activeAnalyserBtn = document.querySelector("#analyser-nav .nav-link.active");
        if (sectionTitle) {
            sectionTitle.textContent = activeAnalyserBtn
                ? activeAnalyserBtn.textContent.trim()
                : "Overview";
        }
        if (sectionIcon) sectionIcon.className = "bi bi-speedometer2 text-primary me-2";
    }

    function activateCodeMode() {
        currentMode = "code";
        codeBtn?.classList.add("active");
        analyserBtn?.classList.remove("active");
        if (analyserMode) analyserMode.style.display = "none";
        if (codeMode) codeMode.style.display = "block";
        analyserNav?.classList.add("d-none");
        codeNav?.classList.remove("d-none");

        if (sectionTitle) sectionTitle.textContent = "Code Notebook";
        if (sectionIcon) sectionIcon.className = "bi bi-code-slash text-primary me-2";

        if (!notebookInitialized) {
            notebookInitialized = true;
            initNotebook();
        }
    }

    analyserBtn?.addEventListener("click", activateAnalyserMode);
    codeBtn?.addEventListener("click", activateCodeMode);

    // =====================================================================
    // 2. Notebook state + helpers
    // =====================================================================
    const cellsContainer = document.getElementById("notebook-cells");
    let cellSeq = 0;
    const editors = {}; // cellId -> CodeMirror instance (or null if unavailable)

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function setDatasetStatus(text, ok) {
        const el = document.getElementById("code-dataset-status");
        if (el) el.textContent = text;
        const icon = el?.parentElement?.querySelector("i");
        if (icon) {
            icon.className = ok
                ? "bi bi-database-check text-success"
                : "bi bi-hourglass-split text-warning";
        }
    }

    function renderPreview(html) {
        const el = document.getElementById("nb-df-preview");
        if (el) el.innerHTML = html || '<div class="text-muted small">No preview available.</div>';
    }

    // =====================================================================
    // 3. Cell creation / rendering
    // =====================================================================
    function makeCellElement(id, code) {
        const wrapper = document.createElement("div");
        wrapper.className = "nb-cell card border-0 shadow-sm mb-3";
        wrapper.dataset.cellId = id;
        wrapper.innerHTML = `
            <div class="nb-cell-row">
                <div class="nb-cell-gutter">
                    <span class="nb-exec-count">[ ]</span>
                    <button class="btn btn-sm btn-link text-secondary nb-run-btn" type="button" title="Run cell (Shift+Enter)">
                        <i class="bi bi-play-circle-fill fs-5"></i>
                    </button>
                    <button class="btn btn-sm btn-link text-secondary nb-delete-btn" type="button" title="Delete cell">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
                <div class="nb-cell-body flex-grow-1">
                    <textarea class="nb-code-input form-control" rows="3" spellcheck="false"
                        placeholder="# Write Python here, e.g. df.describe()">${escapeHtml(code || "")}</textarea>
                    <div class="nb-cell-output"></div>
                </div>
            </div>
        `;
        return wrapper;
    }

    function attachEditor(cellEl, id) {
        const textarea = cellEl.querySelector(".nb-code-input");
        if (typeof CodeMirror === "undefined") {
            editors[id] = null;
            return;
        }
        const cm = CodeMirror.fromTextArea(textarea, {
            mode: "python",
            theme: "dracula",
            lineNumbers: false,
            viewportMargin: Infinity,
            indentUnit: 4,
            tabSize: 4,
            extraKeys: {
                "Shift-Enter": function () { runCell(id, true); },
                "Ctrl-Enter": function () { runCell(id, false); },
                "Cmd-Enter": function () { runCell(id, false); }
            }
        });
        editors[id] = cm;
    }

    function getCellCode(id) {
        const cm = editors[id];
        if (cm) return cm.getValue();
        const cellEl = cellsContainer.querySelector(`[data-cell-id="${id}"]`);
        return cellEl ? cellEl.querySelector(".nb-code-input").value : "";
    }

    function addCell(code, focus) {
        const id = `cell-${++cellSeq}`;
        const cellEl = makeCellElement(id, code);
        cellsContainer.appendChild(cellEl);
        attachEditor(cellEl, id);

        cellEl.querySelector(".nb-run-btn").addEventListener("click", () => runCell(id, false));
        cellEl.querySelector(".nb-delete-btn").addEventListener("click", () => deleteCell(id));

        if (focus) {
            const cm = editors[id];
            if (cm) cm.focus();
            else cellEl.querySelector(".nb-code-input")?.focus();
        }
        return id;
    }

    function deleteCell(id) {
        const cellEl = cellsContainer.querySelector(`[data-cell-id="${id}"]`);
        if (cellEl) cellEl.remove();
        delete editors[id];
        // Notebooks always keep at least one cell to type into.
        if (!cellsContainer.querySelector(".nb-cell")) {
            addCell("", true);
        }
    }

    function orderedCellIds() {
        return Array.from(cellsContainer.querySelectorAll(".nb-cell")).map((el) => el.dataset.cellId);
    }

    // =====================================================================
    // 4. Output rendering
    // =====================================================================
    function renderOutput(id, result) {
        const cellEl = cellsContainer.querySelector(`[data-cell-id="${id}"]`);
        if (!cellEl) return;

        const execCountEl = cellEl.querySelector(".nb-exec-count");
        if (execCountEl) execCountEl.textContent = `[${result.exec_count}]`;

        const outputEl = cellEl.querySelector(".nb-cell-output");
        let html = "";

        if (result.stdout) {
            html += `<pre class="nb-stdout">${escapeHtml(result.stdout)}</pre>`;
        }

        (result.outputs || []).forEach((out) => {
            if (out.type === "html") {
                html += `<div class="nb-output-table table-responsive">${out.content}</div>`;
            } else if (out.type === "image") {
                html += `<img class="nb-output-image" src="data:image/png;base64,${out.content}" alt="plot output">`;
            } else if (out.type === "text") {
                html += `<pre class="nb-stdout">${escapeHtml(out.content)}</pre>`;
            }
        });

        if (result.stderr) {
            html += `<pre class="nb-error">${escapeHtml(result.stderr)}</pre>`;
        }

        if (result.error) {
            html += `<pre class="nb-error">${escapeHtml(result.error)}</pre>`;
        }

        outputEl.innerHTML = html;
        outputEl.classList.toggle("has-content", !!html);
    }

    // =====================================================================
    // 5. Execution (talks to /code/execute, /code/init, /code/reset)
    // =====================================================================
    function runCell(id, advanceToNext) {
        const cellEl = cellsContainer.querySelector(`[data-cell-id="${id}"]`);
        if (!cellEl) return;
        const code = getCellCode(id);

        cellEl.classList.add("nb-cell-running");
        const execCountEl = cellEl.querySelector(".nb-exec-count");
        if (execCountEl) execCountEl.textContent = "[*]";

        return fetch("/code/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code })
        })
            .then((r) => r.json())
            .then((data) => {
                cellEl.classList.remove("nb-cell-running");
                if (data.status === "success") {
                    renderOutput(id, data);
                } else {
                    if (execCountEl) execCountEl.textContent = "[ ]";
                    const outputEl = cellEl.querySelector(".nb-cell-output");
                    outputEl.innerHTML = `<pre class="nb-error">${escapeHtml(data.message || "Execution failed.")}</pre>`;
                    outputEl.classList.add("has-content");
                }

                if (advanceToNext) {
                    const ids = orderedCellIds();
                    const idx = ids.indexOf(id);
                    if (idx === ids.length - 1) {
                        const newId = addCell("", true);
                        return newId;
                    } else {
                        const nextId = ids[idx + 1];
                        const cm = editors[nextId];
                        if (cm) cm.focus();
                    }
                }
            })
            .catch((err) => {
                cellEl.classList.remove("nb-cell-running");
                if (execCountEl) execCountEl.textContent = "[ ]";
                const outputEl = cellEl.querySelector(".nb-cell-output");
                outputEl.innerHTML = `<pre class="nb-error">Server error: ${escapeHtml(String(err))}</pre>`;
                outputEl.classList.add("has-content");
            });
    }

    function runAllCells() {
        const ids = orderedCellIds();
        let chain = Promise.resolve();
        ids.forEach((id) => {
            chain = chain.then(() => runCell(id, false));
        });
        return chain;
    }

    function initNotebook() {
        setDatasetStatus("Preparing dataset…", false);
        fetch("/code/init")
            .then((r) => r.json())
            .then((data) => {
                if (data.status !== "success") {
                    setDatasetStatus(data.message || "No active dataset.", false);
                    return;
                }
                const prepNote = data.prep_steps_pending > 0
                    ? ` (${data.prep_steps_pending} unsaved preprocessing step${data.prep_steps_pending > 1 ? "s" : ""} applied)`
                    : "";
                setDatasetStatus(
                    `df ready — ${data.total_rows.toLocaleString()} rows × ${data.total_cols} cols${prepNote}`,
                    true
                );
                renderPreview(data.preview);

                if (!cellsContainer.querySelector(".nb-cell")) {
                    addCell("# `df` already has your dataset loaded (preprocessing included, if applied).\ndf.head()", true);
                }
            })
            .catch(() => setDatasetStatus("Could not reach the dataset. Try again.", false));
    }

    function restartKernel() {
        if (!confirm("Restart the kernel and re-import the dataset? This clears all variables created in this notebook.")) {
            return;
        }
        setDatasetStatus("Restarting kernel…", false);
        fetch("/code/reset", { method: "POST" })
            .then((r) => r.json())
            .then((data) => {
                if (data.status !== "success") {
                    setDatasetStatus(data.message || "Restart failed.", false);
                    return;
                }
                setDatasetStatus(
                    `df re-imported — ${data.total_rows.toLocaleString()} rows × ${data.total_cols} cols`,
                    true
                );
                renderPreview(data.preview);

                // Clear outputs + exec counts on every existing cell, but keep the code.
                cellsContainer.querySelectorAll(".nb-cell").forEach((cellEl) => {
                    cellEl.querySelector(".nb-exec-count").textContent = "[ ]";
                    const outputEl = cellEl.querySelector(".nb-cell-output");
                    outputEl.innerHTML = "";
                    outputEl.classList.remove("has-content");
                });
            })
            .catch(() => setDatasetStatus("Server error while restarting.", false));
    }

    // =====================================================================
    // 6. Toolbar + sidebar wiring
    // =====================================================================
    document.getElementById("nb-add-cell")?.addEventListener("click", () => addCell("", true));
    document.getElementById("nb-run-all")?.addEventListener("click", runAllCells);
    document.getElementById("nb-restart-kernel")?.addEventListener("click", restartKernel);

    document.getElementById("nb-sidebar-add-cell")?.addEventListener("click", () => addCell("", true));
    document.getElementById("nb-sidebar-run-all")?.addEventListener("click", runAllCells);
    document.getElementById("nb-sidebar-restart")?.addEventListener("click", restartKernel);
});
