document.addEventListener("DOMContentLoaded", function () {
    // ==========================================
    // 1. Sidebar Tab Navigation
    // ==========================================
    const navButtons = document.querySelectorAll(".sidebar .nav-link");
    const sections = document.querySelectorAll(".content-section");

    navButtons.forEach(button => {
        button.addEventListener("click", function () {
            const sectionId = this.dataset.section;

            // Hide all sections
            sections.forEach(s => s.style.display = "none");

            // Show active section
            const target = document.getElementById(sectionId);
            if (target) {
                target.style.display = "block";
            }

            // Update active state on sidebar
            navButtons.forEach(btn => btn.classList.remove("active"));
            this.classList.add("active");
        });
    });

    // ==========================================
    // 2. Data Preparation Sub-Navigation Switching
    // ==========================================
    document.addEventListener("click", function (e) {
        const subNavBtn = e.target.closest("[data-prep-target]");
        if (subNavBtn) {
            document.querySelectorAll("#prepNavTabs .nav-link").forEach(b => b.classList.remove("active"));
            subNavBtn.classList.add("active");

            const targetId = subNavBtn.getAttribute("data-prep-target");
            document.querySelectorAll(".prep-tab-content").forEach(c => c.style.display = "none");

            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.style.display = "block";
            }
        }
    });

    // ==========================================
    // 3. Prep Metadata & UI Refresh Helpers
    // ==========================================
    function resetPreprocessingUI() {
        // Disable Undo and Save buttons
        const undoBtn = document.getElementById("btn-undo-replace");
        const saveBtn = document.getElementById("btn-save-replace");
        if (undoBtn) undoBtn.disabled = true;
        if (saveBtn) saveBtn.disabled = true;

        // Hide status banners/alerts
        const statusAlert = document.getElementById("replace-status-alert");
        if (statusAlert) statusAlert.classList.add("d-none");

        // Clear input values
        const findInput = document.getElementById("find-val-input");
        const replaceInput = document.getElementById("replace-val-input");
        if (findInput) findInput.value = "";
        if (replaceInput) replaceInput.value = "";
    }

    function refreshPrepMetadata() {
        fetch("/prep/info")
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    updatePrepUI(data);
                }
            })
            .catch(err => console.error("Error fetching prep metadata:", err));
    }

    function updatePrepUI(data) {
        const totalRowsEl = document.getElementById("prep-total-rows");
        const totalColsEl = document.getElementById("prep-total-cols");
        const dupCountEl = document.getElementById("dup-count-display");
        const missingTbody = document.getElementById("missing-info-tbody");

        if (totalRowsEl) totalRowsEl.innerText = data.total_rows;
        if (totalColsEl) totalColsEl.innerText = data.total_cols;
        if (dupCountEl) dupCountEl.innerText = `${data.duplicates} Rows`;

        // Populate df.info() tbody
        if (missingTbody && data.info) {
            let html = "";
            data.info.forEach(row => {
                html += `<tr>
                    <td><code>${row.column}</code></td>
                    <td>${row.non_null}</td>
                    <td><span class="${row.null_count > 0 ? 'text-danger fw-bold' : 'text-muted'}">${row.null_count}</span></td>
                    <td><span class="badge bg-light text-dark border">${row.dtype}</span></td>
                </tr>`;
            });
            missingTbody.innerHTML = html;
        }

        // 2. Encoding Dropdown — Populate ONLY with categorical columns
        const encodeSelect = document.getElementById("encode-col-select");
        if (encodeSelect && data.info) {
            const categoricalCols = data.info.filter(col => col.is_categorical);

            let optionsHtml = '<option value="">-- Select Categorical Column --</option>';
            categoricalCols.forEach(col => {
                optionsHtml += `<option value="${col.column}">${col.column} (${col.dtype})</option>`;
            });

            encodeSelect.innerHTML = optionsHtml;
        }

        // If a column is currently selected in Type Modify, refresh its displayed dtype
        const typeColSelect = document.getElementById("type-col-select");
        if (typeColSelect && typeColSelect.value) {
            updateCurrentDtypeDisplay(typeColSelect.value, data.info);
        }
    }

    function executePrepAction(payload) {
        fetch("/prep/apply", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(r => r.json())
            .then(data => {
                if (data.status === "success") {
                    alert(data.message || "Action applied successfully.");

                    const previewEl = document.getElementById("prep-dataset-preview");
                    if (previewEl && data.preview) {
                        previewEl.innerHTML = data.preview;
                    }

                    // Update dataset preview tables across the rest of dashboard
                    document.querySelectorAll(".dataset-preview-container").forEach(el => {
                        if (data.preview) el.innerHTML = data.preview;
                    });

                    // Update prep UI and sync new metadata
                    updatePrepUI(data);
                    refreshPrepMetadata();
                } else {
                    alert("Error: " + (data.error || "Execution failed."));
                }
            })
            .catch(err => alert("Server error: " + err));
    }

    // Dynamic dropdown for missing values strategy
    const missingColSelect = document.getElementById("missing-col-select");
    const missingActionSelect = document.getElementById("missing-action-select");

    if (missingColSelect && missingActionSelect) {
        missingColSelect.addEventListener("change", function () {
            const col = this.value;
            if (!col) {
                missingActionSelect.disabled = true;
                missingActionSelect.innerHTML = `<option>-- Select Column First --</option>`;
                return;
            }

            fetch("/prep/info")
                .then(r => r.json())
                .then(d => {
                    const colInfo = d.info ? d.info.find(i => i.column === col) : null;
                    let opts = `<option value="drop_col" class="text-danger fw-bold">🗑️ Drop Column Entirely</option>
                                <option value="drop">Drop Rows with Nulls</option>
                                <option value="mode">Fill with Mode (Most Frequent)</option>
                                <option value="ffill">Forward Fill (ffill)</option>
                                <option value="bfill">Backward Fill (bfill)</option>`;

                    if (colInfo && colInfo.is_numeric) {
                        opts += `<option value="mean">Fill with Mean</option>
                                 <option value="median">Fill with Median</option>`;
                    }
                    missingActionSelect.innerHTML = opts;
                    missingActionSelect.disabled = false;
                })
                .catch(err => console.error("Error loading column info:", err));
        });
    }

    // Dynamic Display of Current Data Type in Type Modify Tab
    const typeColSelect = document.getElementById("type-col-select");
    const typeCurrentDisplay = document.getElementById("type-current-display");
    const currentTypeBadge = document.getElementById("current-type-badge");

    function updateCurrentDtypeDisplay(selectedCol, infoList) {
        if (!typeCurrentDisplay || !currentTypeBadge) return;

        if (!selectedCol) {
            typeCurrentDisplay.style.display = "none";
            return;
        }

        const renderDtype = (colInfo) => {
            if (colInfo) {
                currentTypeBadge.textContent = colInfo.dtype;
                typeCurrentDisplay.style.display = "block";
            }
        };

        if (infoList) {
            const colInfo = infoList.find(i => i.column === selectedCol);
            renderDtype(colInfo);
        } else {
            fetch("/prep/info")
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success" && data.info) {
                        const colInfo = data.info.find(i => i.column === selectedCol);
                        renderDtype(colInfo);
                    }
                })
                .catch(err => console.error("Error fetching column data type:", err));
        }
    }

    if (typeColSelect) {
        typeColSelect.addEventListener("change", function () {
            updateCurrentDtypeDisplay(this.value);
        });
    }

    // ==========================================
    // 4. Data Preparation Action Handlers
    // ==========================================
    document.getElementById("btn-apply-missing")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({
            action: "missing",
            column: missingColSelect ? missingColSelect.value : "",
            strategy: missingActionSelect ? missingActionSelect.value : ""
        });
    });

    document.getElementById("btn-drop-duplicates")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({ action: "drop_duplicates" });
    });

    document.getElementById("btn-apply-type")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({
            action: "type_modify",
            column: document.getElementById("type-col-select")?.value,
            target_type: document.getElementById("type-target-select")?.value
        });
    });

    // Uniqueness Inspector
    document.getElementById("btn-inspect-unique")?.addEventListener("click", (e) => {
        e.preventDefault();
        const col = document.getElementById("unique-col-select")?.value;
        if (!col) return alert("Select a column first.");

        fetch("/prep/inspect-unique", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ column: col })
        })
            .then(r => r.json())
            .then(res => {
                if (res.status === "success") {
                    const d = res.data;
                    document.getElementById("unique-count").innerText = d.unique_count;
                    document.getElementById("unique-total-rows").innerText = d.total_rows;

                    let rows = "";
                    d.values.forEach(v => {
                        rows += `<tr><td><code>${v.value}</code></td><td>${v.count}</td></tr>`;
                    });
                    document.getElementById("unique-values-tbody").innerHTML = rows;
                    document.getElementById("unique-results-box").style.display = "block";
                } else {
                    alert("Inspection failed: " + (res.error || "Unknown error"));
                }
            })
            .catch(err => alert("Server error: " + err));
    });

    document.getElementById("btn-apply-encode")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({
            action: "encode",
            column: document.getElementById("encode-col-select")?.value,
            method: document.getElementById("encode-method-select")?.value
        });
    });

    document.getElementById("btn-apply-outlier")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({
            action: "outliers",
            column: document.getElementById("outlier-col-select")?.value
        });
    });

    document.getElementById("btn-apply-scale")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({
            action: "scale",
            column: document.getElementById("scale-col-select")?.value,
            method: document.getElementById("scale-method-select")?.value
        });
    });

    document.getElementById("btn-apply-text-clean")?.addEventListener("click", (e) => {
        e.preventDefault();
        executePrepAction({
            action: "clean_text",
            column: document.getElementById("text-col-select")?.value
        });
    });

    // Legacy Preprocessing Form Handler (if present in template)
    const preprocessForm = document.getElementById("preprocessing-form");
    if (preprocessForm) {
        preprocessForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const payload = {
                missing_strategy: document.getElementById("missing_strategy")?.value,
                encode_categories: document.getElementById("encode_categories")?.checked,
                scale_numeric: document.getElementById("scale_numeric")?.checked
            };

            fetch("/apply-preprocessing", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            })
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success") {
                        document.querySelectorAll(".dataset-preview-container").forEach(el => {
                            el.innerHTML = data.preview;
                        });

                        const alertBox = document.getElementById("preprocessing-alert");
                        if (alertBox) {
                            alertBox.className = "alert alert-success mt-3";
                            alertBox.textContent = data.message;
                            alertBox.classList.remove("d-none");
                        }
                    } else {
                        alert("Preprocessing error: " + data.error);
                    }
                })
                .catch(err => console.error("Error:", err));
        });
    }

    // --- HELPER FUNCTIONS FOR FIND & REPLACE / UNDO ---

    function toggleReplaceControlState(undoAvailable, message) {
        const undoBtn = document.getElementById("btn-undo-replace");
        const saveBtn = document.getElementById("btn-save-replace");
        const alertBox = document.getElementById("replace-status-alert");
        const alertMsg = document.getElementById("replace-status-msg");

        if (undoBtn) undoBtn.disabled = !undoAvailable;
        if (saveBtn) saveBtn.disabled = !undoAvailable;

        if (alertBox && alertMsg && message) {
            alertBox.classList.remove("d-none");
            alertMsg.textContent = message;
        }
    }

    function reinspectActiveUniqueColumn() {
        const select = document.getElementById("unique-col-select");
        const inspectBtn = document.getElementById("btn-inspect-unique");

        if (select && select.value && inspectBtn) {
            inspectBtn.click(); // Trigger re-inspection automatically
        }
    }

    // --- FIND & REPLACE, UNDO, SAVE EVENT LISTENERS ---

    // 1. APPLY REPLACE
    document.getElementById("btn-apply-replace")?.addEventListener("click", (e) => {
        e.preventDefault();
        const col = document.getElementById("unique-col-select")?.value;
        const findVal = document.getElementById("find-val-input")?.value;
        const replaceVal = document.getElementById("replace-val-input")?.value;

        if (!col) return alert("Please select a column first.");
        if (findVal === "" || findVal === undefined) return alert("Please enter a value to find.");

        fetch("/prep/apply", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                action: "replace",
                column: col,
                find: findVal,
                replace: replaceVal
            })
        })
            .then(r => r.json())
            .then(data => {
                if (data.status === "success") {
                    toggleReplaceControlState(data.undo_available, data.message);

                    // Refresh preview if container exists
                    const previewContainer = document.getElementById("prep-dataset-preview");
                    if (previewContainer && data.preview) {
                        previewContainer.innerHTML = data.preview;
                    }

                    reinspectActiveUniqueColumn();
                } else {
                    alert("Error: " + (data.error || "Replacement failed."));
                }
            })
            .catch(err => alert("Server error: " + err));
    });

    // 2. UNDO LAST OPERATION
    document.getElementById("btn-undo-replace")?.addEventListener("click", (e) => {
        e.preventDefault();

        fetch("/prep/apply", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "undo" })
        })
            .then(r => r.json())
            .then(data => {
                if (data.status === "success") {
                    toggleReplaceControlState(data.undo_available, data.message);

                    // Refresh preview if container exists
                    const previewContainer = document.getElementById("prep-dataset-preview");
                    if (previewContainer && data.preview) {
                        previewContainer.innerHTML = data.preview;
                    }

                    reinspectActiveUniqueColumn();
                } else {
                    alert("Error: " + (data.error || "Undo failed."));
                }
            })
            .catch(err => alert("Server error: " + err));
    });

    // 3. SAVE ALL CHANGES PERMANENTLY TO DISK
    document.getElementById("btn-save-replace")?.addEventListener("click", (e) => {
        e.preventDefault();

        fetch("/prep/apply", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "save" })
        })
            .then(r => r.json())
            .then(data => {
                if (data.status === "success") {
                    toggleReplaceControlState(false, data.message);
                    alert("Dataset successfully saved to disk!");

                    const previewContainer = document.getElementById("prep-dataset-preview");
                    if (previewContainer && data.preview) {
                        previewContainer.innerHTML = data.preview;
                    }
                } else {
                    alert("Error: " + (data.error || "Save failed."));
                }
            })
            .catch(err => alert("Server error: " + err));
    });

    // ==========================================
    // 5. Customer Segmentation Form Handling (AJAX)
    // ==========================================
    const segmentationForm = document.getElementById("segmentation-form");
    if (segmentationForm) {
        segmentationForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const payload = {
                id_col: document.getElementById("id_col")?.value,
                date_col: document.getElementById("date_col")?.value,
                amount_col: document.getElementById("amount_col")?.value
            };

            fetch("/run-segmentation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            })
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success") {
                        document.getElementById("rfm-results-container").innerHTML = data.html_table;
                        document.getElementById("segmentation-summary")?.classList.remove("d-none");
                    } else {
                        alert("Segmentation failed: " + data.error);
                    }
                })
                .catch(err => console.error("Error:", err));
        });
    }

    // ==========================================
    // 6. Recommendation Form Handling (AJAX)
    // ==========================================
    const recForm = document.getElementById("recommendation-form");
    if (recForm) {
        recForm.addEventListener("submit", function (e) {
            e.preventDefault();
            fetch("/run-recommendation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_col: document.getElementById("rec_user_col")?.value,
                    item_col: document.getElementById("rec_item_col")?.value,
                    qty_col: document.getElementById("rec_qty_col")?.value
                })
            })
                .then(res => res.json())
                .then(res => {
                    if (res.status === "success") {
                        const popList = document.getElementById("popular-items-list");
                        if (popList && res.data.popular_items) {
                            popList.innerHTML = res.data.popular_items.map(i => `<li class="list-group-item badge bg-primary me-2 mb-2 fs-6">${i}</li>`).join("");
                        }

                        const tbody = document.getElementById("user-recs-tbody");
                        if (tbody && res.data.user_recommendations) {
                            tbody.innerHTML = Object.entries(res.data.user_recommendations).map(([u, items]) =>
                                `<tr><td><strong>${u}</strong></td><td>${items.join(", ")}</td></tr>`
                            ).join("");
                        }

                        document.getElementById("recommendation-results")?.classList.remove("d-none");
                    } else {
                        alert("Recommendation error: " + (res.error || "Failed to calculate recommendations."));
                    }
                })
                .catch(err => console.error("Error:", err));
        });
    }

    // NOTE: Review Sentiment form submission is handled entirely by the
    // inline script in templates/sections/reviews.html (handleSentimentSubmit),
    // which is kept in sync with the current /run-reviews response shape
    // (global.positive.pct/count, product_breakdown[].positive_str, etc.).
    // A second, outdated handler used to be registered here as well; having
    // two "submit" listeners on the same #review-form meant every submit
    // fired /run-reviews twice, and whichever response resolved last would
    // overwrite the DOM -- sometimes with the correct result, sometimes with
    // this stale handler's mismatched field names (undefined / [object
    // Object]%). That duplicate handler has been removed.

    // Initial setup calls on application start
    resetPreprocessingUI();
    refreshPrepMetadata();
});