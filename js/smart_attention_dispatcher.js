/**
 * smart_attention_dispatcher.js
 * ------------------------------
 * v1.0.6
 * UI for SmartAttentionDispatcher:
 * - Instant UI updates via WebSocket.
 * - Properly hidden widget to persist status across page reloads.
 * - Restores status on F5 via onConfigure hook.
 * - Color coding for different modes (SA2, SA3, Dynamic, SDPA).
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "SmartAttentionDispatcher";
const MIN_WIDTH = 480;
const LINE_H    = 24;   // px — row height
const PADDING   = 20;   // px — vertical padding

function _ensureStyle() {
    const ID = "rogala-sad-style";
    if (document.getElementById(ID)) return;

    const style = document.createElement("style");
    style.id    = ID;
    style.textContent = `
        .sad-panel {
            background    : var(--comfy-input-bg, #1a1f1c);
            border        : 1px solid var(--border-color, #444);
            border-radius : 4px;
            font-family   : "Fira Code", "Cascadia Code", monospace;
            font-size     : 14px;
            line-height   : 1.6em;
            color         : var(--fg-color, #c8e8c8);
            padding       : 8px 12px;
            width         : 100%;
            height        : 100%;
            box-sizing    : border-box;
            overflow      : hidden;
            white-space   : pre;
            user-select   : text;
        }
        .sad-placeholder { color: #666; font-style: italic; }
        
        .sad-sdpa    { color: #e0a84a; font-weight: bold; }
        .sad-sa2     { color: #6dbf67; font-weight: bold; }
        .sad-sa3     { color: #5fbcf9; font-weight: bold; }
        .sad-dynamic { color: #c586c0; font-weight: bold; }
        
        .sad-tier    { color: #9cdcfe; font-style: italic; }
        .sad-ok      { color: #6dbf67; }
        .sad-warn    { color: #e0a84a; }
        .sad-err     { color: #cf5f5f; }
        .sad-dim     { color: #888; }
    `;
    document.head.appendChild(style);
}

function _renderStatus(text) {
    if (!text) {
        return `<span class="sad-placeholder">Run the node to see detection status.</span>`;
    }

    const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    let html = escape(text);

    html = html.replace(
        /^(Mode:\s*)(.*)$/m,
        (_, prefix, rest) => {
            let parts = rest.split("&gt;&gt;&gt;");
            let mainPart = parts[0].trim();
            let fallbackPart = parts.length > 1 ? parts[1].trim() : null;

            const getColorClass = (str) => {
                if (str.includes("SA2-SA3") || str.includes("SDPA-SA3")) return "sad-dynamic";
                if (str.includes("SA3")) return "sad-sa3";
                if (str.includes("SA2")) return "sad-sa2";
                if (str.includes("SDPA")) return "sad-sdpa";
                if (str.includes("ERROR")) return "sad-err";
                return "sad-sdpa";
            };

            let mainHtml = `<span class="${getColorClass(mainPart)}">${mainPart}</span>`;
            if (fallbackPart) {
                return `${prefix}${mainHtml} <span class="sad-dim">&gt;&gt;&gt;</span> <span class="${getColorClass(fallbackPart)}">${fallbackPart}</span>`;
            }
            return `${prefix}${mainHtml}`;
        }
    );

    html = html.replace(/\b(Blackwell|Blackwell DC|Hopper|Ada|Ampere|Turing)\b/g, '<span class="sad-tier">$1</span>');
    html = html.replace(/\bOK\b/g, '<span class="sad-ok">OK</span>');
    html = html.replace(/\b--\b/g, '<span class="sad-err">--</span>');
    html = html.replace(/\|/g, '<span class="sad-dim">|</span>');
    html = html.replace(/^(Note:.+)$/m, '<span class="sad-warn">$1</span>');

    return html;
}

app.registerExtension({
    name: "rogala.SmartAttentionDispatcher",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        _ensureStyle();

        const _onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _onNodeCreated?.apply(this, arguments);

            let statusWidget = this.widgets?.find(w => w.name === "sad_status_str");
            if (!statusWidget) {
                statusWidget = this.addWidget("string", "sad_status_str", "", () => {});
                statusWidget.hidden = true;
            }
            this._statusWidget = statusWidget;

            const panel = document.createElement("div");
            panel.className = "sad-panel";
            panel.innerHTML = _renderStatus(statusWidget.value);
            this._sadPanel = panel;

            this.addDOMWidget("sad_status", "div", panel, {
                serialize    : false,
                hideOnZoom   : false,
                getMinHeight : () => LINE_H * 5 + PADDING,
                getMaxHeight : () => LINE_H * 10 + PADDING,
                getHeight    : () => {
                    const lines = (this._sadPanel?.innerText || "").split("\n").filter(Boolean).length || 5;
                    return Math.max(5, lines) * LINE_H + PADDING;
                },
            });

            const _computeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function (w) {
                const base = _computeSize?.apply(this, arguments) ?? [MIN_WIDTH, 240];
                base[0] = Math.max(base[0], MIN_WIDTH);
                return base;
            };

            // WebSocket listener — bound here so 'this' correctly refers to the node instance
            const nodeId = this.id;
            const self   = this;
            api.addEventListener("rogala/sad_status", (e) => {
                if (e.detail.node !== nodeId.toString()) return;
                if (self._sadPanel) {
                    self._sadPanel.innerHTML = _renderStatus(e.detail.text);
                }
                if (self._statusWidget) {
                    self._statusWidget.value = e.detail.text;
                }
                self.setSize(self.computeSize());
                self.setDirtyCanvas(true);
            });
        };

        // Restore status when node is loaded from saved workflow (e.g. F5 reload)
        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            _onConfigure?.apply(this, arguments);
            if (this._statusWidget && this._sadPanel) {
                this._sadPanel.innerHTML = _renderStatus(this._statusWidget.value);
            }
        };

        const _onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            _onExecuted?.apply(this, arguments);
            const text = message?.text?.[0] ?? "";
            if (this._sadPanel) {
                this._sadPanel.innerHTML = _renderStatus(text);
            }
            if (this._statusWidget) {
                this._statusWidget.value = text;
            }
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        const _onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            _onRemoved?.apply(this, arguments);
            if (this._sadPanel) {
                this._sadPanel.remove();
                this._sadPanel = null;
            }
        };
    },
});