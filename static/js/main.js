/* ===================================================================
   Traffic Volume Prediction System — Main JavaScript
   =================================================================== */

/**
 * Format a number with commas and zero decimal places.
 * @param {number} num - The number to format.
 * @returns {string} The formatted number string (e.g. "4,523").
 */
function formatNumber(num) {
    if (num == null || isNaN(num)) return "—";
    return Math.round(num).toLocaleString("en-US", {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    });
}

/**
 * Determine the traffic level category based on predicted volume.
 *
 * Thresholds:
 *   < 2000      → Low
 *   2000 - 4500 → Medium
 *   > 4500      → High
 *
 * @param {number} volume - Predicted traffic volume (veh/h).
 * @returns {{level: string, class: string, label: string}} Traffic level metadata.
 */
function getTrafficLevel(volume) {
    if (volume < 2000) {
        return {
            level: "Low",
            class: "level-low",
            label: "Low Traffic (< 2,000 veh/h)",
        };
    } else if (volume < 4500) {
        return {
            level: "Medium",
            class: "level-medium",
            label: "Medium Traffic (2,000 – 4,500 veh/h)",
        };
    } else {
        return {
            level: "High",
            class: "level-high",
            label: "High Traffic (> 4,500 veh/h)",
        };
    }
}

// ---------------------------------------------------------------------------
// DOM Initializations
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", function () {
    // 1. Mobile navigation menu toggle
    const navToggle = document.getElementById("navToggle");
    const navLinks = document.getElementById("navLinks");

    if (navToggle && navLinks) {
        navToggle.addEventListener("click", function () {
            navLinks.classList.toggle("open");
            navToggle.classList.toggle("active");
        });
    }

    // 2. Auto-set weekend checkbox when day of week changes
    const daySelect = document.getElementById("day_of_week");
    const weekendCheckbox = document.getElementById("is_weekend");

    if (daySelect && weekendCheckbox) {
        daySelect.addEventListener("change", function () {
            const day = parseInt(daySelect.value, 10);
            // Saturday = 5, Sunday = 6
            weekendCheckbox.checked = (day === 5 || day === 6);
        });
    }
});
