/* ============================================
   Traffic Volume Prediction - Main JavaScript
   ============================================ */

/**
 * Format a number with commas and zero decimal places.
 * @param {number} num - The number to format.
 * @returns {string} The formatted number string (e.g. "4,523").
 */
function formatNumber(num) {
    return Math.round(num).toLocaleString("en-US", {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    });
}

/**
 * Show the loading spinner and hide the results container.
 */
function showLoading() {
    const spinner = document.getElementById("loading-spinner");
    const results = document.getElementById("prediction-results");

    if (spinner) spinner.classList.remove("hidden");
    if (results) results.classList.add("hidden");
}

/**
 * Hide the loading spinner.
 */
function hideLoading() {
    const spinner = document.getElementById("loading-spinner");
    if (spinner) spinner.classList.add("hidden");
}

/**
 * Display an error message inside the results container.
 * @param {string} message - The error message to display.
 */
function showError(message) {
    hideLoading();

    const results = document.getElementById("prediction-results");
    if (results) {
        results.classList.remove("hidden");
        results.innerHTML =
            '<div class="alert alert-error">' +
            "<strong>Error:</strong> " +
            message +
            "</div>";
    }
}

/**
 * Determine the traffic level category based on predicted volume.
 *
 * Thresholds:
 *   0   – 2000  → Low
 *   2000 – 4500 → Medium
 *   4500+        → High
 *
 * @param {number} volume - Predicted traffic volume.
 * @returns {{level: string, class: string}} Traffic level label and CSS class.
 */
function getTrafficLevel(volume) {
    if (volume < 2000) {
        return { level: "Low", class: "traffic-low" };
    } else if (volume < 4500) {
        return { level: "Medium", class: "traffic-medium" };
    } else {
        return { level: "High", class: "traffic-high" };
    }
}

/* ----- Auto-set weekend checkbox based on day_of_week ----- */
document.addEventListener("DOMContentLoaded", function () {
    const daySelect = document.getElementById("day_of_week");
    const weekendCheckbox = document.getElementById("is_weekend");

    if (daySelect && weekendCheckbox) {
        daySelect.addEventListener("change", function () {
            const day = parseInt(daySelect.value, 10);
            // Saturday = 5, Sunday = 6 (Monday-indexed: 0 = Monday)
            weekendCheckbox.checked = day >= 5;
        });
    }
});
