document.addEventListener('DOMContentLoaded', function() {
    const dropdown = document.getElementById('comp-dropdown');
    if (dropdown) {
        dropdown.addEventListener('change', function() {
            const selectedComp = this.value.replace(/\//g, '---');
            window.location.href = `/refresh/${selectedComp}`;
        });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const table = document.getElementById('leaderboard');
    if (table) {
        const body = table.tBodies[0];
        const rows = Array.from(body.rows);
        table.querySelectorAll('[data-sort-key]').forEach(button => {
            button.addEventListener('click', () => {
                const heading = button.closest('th');
                const previous = heading.getAttribute('aria-sort');
                const direction = previous === 'ascending' ? 'desc' : previous === 'descending' ? 'asc' : (button.dataset.defaultDirection || 'asc');
                const key = button.dataset.sortKey;
                const sorted = [...rows].sort((a, b) => {
                    const left = a.dataset[key];
                    const right = b.dataset[key];
                    if (left === '' || right === '') return left === right ? 0 : left === '' ? 1 : -1;
                    const difference = button.dataset.sortType === 'text'
                        ? left.localeCompare(right, undefined, { sensitivity: 'base', numeric: true })
                        : Number(left) - Number(right);
                    return (direction === 'asc' ? difference : -difference) || a.dataset.model.localeCompare(b.dataset.model);
                });
                sorted.forEach(row => body.appendChild(row));
                table.querySelectorAll('thead th[aria-sort]').forEach(th => {
                    th.setAttribute('aria-sort', 'none');
                    th.querySelector('.sort-indicator').textContent = '↕';
                });
                heading.setAttribute('aria-sort', direction === 'asc' ? 'ascending' : 'descending');
                button.querySelector('.sort-indicator').textContent = direction === 'asc' ? '↑' : '↓';
            });
        });
    }
    document.querySelectorAll('[data-copy-table]').forEach(button => {
        button.addEventListener('click', async () => {
            const table = document.getElementById(button.dataset.copyTable);
            const text = Array.from(table.rows).filter(row => !row.hidden).map(row =>
                Array.from(row.cells).map(cell => {
                    const clone = cell.cloneNode(true);
                    clone.querySelectorAll('.sort-indicator, .row-arrow').forEach(node => node.remove());
                    return clone.textContent.replace(/\s+/g, ' ').trim();
                }).join('\t')
            ).join('\n');
            try {
                await navigator.clipboard.writeText(text);
                button.textContent = 'Copied';
            } catch (_) {
                button.textContent = 'Copy unavailable';
            }
            setTimeout(() => { button.textContent = 'Copy table'; }, 1800);
        });
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const current = document.querySelector('.sidebar-item.current');
    if (current) {
        current.scrollIntoView({ behavior: 'auto', block: 'center' });
    }
    renderMarkedElements(document);
});

function competitionUrl(path) {
    const url = new URL(path, window.location.origin);
    const comp = document.getElementById('comp-dropdown')?.value;
    if (comp) url.searchParams.set('comp', comp);
    return url;
}

function loadResponseBox(element) {
    const panel = element.closest('.run-tabcontent');
    if (panel && panel.style.display === 'none') return;
    if (!element.open || element.hasAttribute('data-loaded') || element.hasAttribute('data-loading')) return;
    element.setAttribute('data-loading', true);
    const idd = element.getAttribute('id');
    const url = competitionUrl(`/modelinteraction/${encodeURIComponent(idd)}`);
    if (element.hasAttribute('data-trace-only')) url.searchParams.set('trace_only', '1');
    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error('Failed to load conversation.');
            return response.text();
        })
        .then(data => {
            const wrapper = document.createElement('div');
            wrapper.className = 'conversation-content';
            wrapper.innerHTML = data;
            element.appendChild(wrapper);
            element.setAttribute('data-loaded', true);
            renderMarkedElements(wrapper);
            if (window.hljs) {
                wrapper.querySelectorAll('pre code:not([data-highlighted])').forEach(block => hljs.highlightElement(block));
            }
        })
        .catch(error => console.error('Error fetching details:', error))
        .finally(() => element.removeAttribute('data-loading'));
}

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.response-box-details').forEach(function(element) {
        element.addEventListener('toggle', function(event) {
            loadResponseBox(event.target);
        });
        loadResponseBox(element);
    });
});

function loadHistoryStep(stepId) {
    if (!stepId) {
        for (const element of document.getElementsByClassName('history-step-content')) {
            element.innerHTML = '';
        }
        return;
    }

    const parts = stepId.split(">>");
    const index = parts[2];

    fetch(competitionUrl(`/historystep/${encodeURIComponent(stepId)}`))
        .then(response => response.text())
        .then(data => {
            const element = document.getElementById(`history-step-content-${index}`);
            element.innerHTML = data;
            renderMarkedElements(element);
            if (window.hljs) {
                element.querySelectorAll('pre code:not([data-highlighted])').forEach(block => hljs.highlightElement(block));
            }
        })
        .catch(error => {
            console.error('Error fetching step:', error);
            document.getElementById(`history-step-content-${index}`).innerHTML = '<div class="error">Error loading step</div>';
        });
}

function syncNavigation() {
    const mobile = window.matchMedia('(max-width: 800px)').matches;
    const open = mobile ? document.body.classList.contains('nav-open') : !document.body.classList.contains('sidebar-collapsed');
    document.getElementById('sidebar-toggle-button')?.setAttribute('aria-expanded', String(open));
}

function toggleSidebar() {
    const mobile = window.matchMedia('(max-width: 800px)').matches;
    document.body.classList.toggle(mobile ? 'nav-open' : 'sidebar-collapsed');
    syncNavigation();
    if (mobile && document.body.classList.contains('nav-open')) {
        document.getElementById('comp-dropdown').focus();
    } else {
        document.getElementById('sidebar-toggle-button').focus();
    }
}

document.addEventListener('DOMContentLoaded', syncNavigation);
window.matchMedia('(max-width: 800px)').addEventListener('change', () => {
    document.body.classList.remove('nav-open');
    syncNavigation();
});
document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('nav-open')) toggleSidebar();
});

function rerenderMath(element, attempt = 0) {
    if (!element) return;
    if (!window.renderMathInElement) {
        if (attempt < 20) {
            setTimeout(() => rerenderMath(element, attempt + 1), 50);
        }
        return;
    }
    renderMathInElement(element, {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false
    });
}

function protectMathBlocks(text) {
    const blocks = [];
    const protectedText = text.replace(/\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$(?:\\.|[^$\\\n])+\$/g, (match) => {
        const token = `CODEXMATHPLACEHOLDER${blocks.length}X`;
        blocks.push(match);
        return token;
    });
    return { protectedText, blocks };
}

const allowedHtmlTags = new Set([
    'a', 'b', 'blockquote', 'br', 'code', 'del', 'details', 'div', 'em',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'li', 'mark',
    'ol', 'p', 'pre', 's', 'small', 'span', 'strong', 'sub', 'summary',
    'sup', 'table', 'tbody', 'td', 'th', 'thead', 'tr', 'u', 'ul'
]);

function protectHtmlTags(text) {
    const tags = [];
    const protectedText = text.replace(/<\/?([A-Za-z][\w-]*)\b[^>]*>/g, (match, tagName) => {
        if (!allowedHtmlTags.has(String(tagName).toLowerCase())) {
            return match;
        }
        const token = `CODEXHTMLPLACEHOLDER${tags.length}X`;
        tags.push(match);
        return token;
    });
    return { protectedText, tags };
}

function escapePlainText(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function restoreHtmlTags(text, tags) {
    return text.replace(/CODEXHTMLPLACEHOLDER(\d+)X/g, (_, index) => tags[Number(index)] ?? '');
}

function restoreMathPlaceholders(text) {
    return text.replace(
        /CODEXMATHPLACEHOLDER(\d+)X/g,
        (_, index) => `<span class="codex-math-placeholder" data-math-index="${index}"></span>`
    );
}

function normalizeMarkdownSource(text) {
    return text.replace(/```xml\b[^\n\r]*\r?\n([\s\S]*?)\r?\n```/gi, '$1');
}

function renderMarkedText(element, text = null) {
    if (!element) return;
    const source = normalizeMarkdownSource(text ?? element.dataset.markdownSource ?? element.textContent ?? '');
    element.dataset.markdownSource = source;
    if (window.marked && typeof window.marked.parse === 'function') {
        const { protectedText: mathProtected, blocks } = protectMathBlocks(source);
        const { protectedText: htmlProtected, tags } = protectHtmlTags(mathProtected);
        const rendered = window.marked.parse(escapePlainText(htmlProtected), { breaks: true });
        element.innerHTML = restoreMathPlaceholders(restoreHtmlTags(rendered, tags));
        element.querySelectorAll('.codex-math-placeholder').forEach((placeholder) => {
            const index = Number(placeholder.dataset.mathIndex);
            placeholder.replaceWith(document.createTextNode(blocks[index] ?? ''));
        });
    } else {
        element.textContent = source;
    }
    rerenderMath(element);
    if (window.hljs) {
        element.querySelectorAll('pre code').forEach((block) => hljs.highlightElement(block));
    }
}

function renderMarkedElements(root = document) {
    root.querySelectorAll('.marked').forEach((element) => {
        if (element.tagName === 'PRE') return;
        renderMarkedText(element);
    });
}

function getScrollableAncestors(element) {
    const ancestors = [];
    let current = element.parentElement;
    while (current) {
        if (current.scrollHeight > current.clientHeight || current.scrollWidth > current.clientWidth) {
            ancestors.push([current, current.scrollTop, current.scrollLeft]);
        }
        current = current.parentElement;
    }
    ancestors.push([window, window.scrollY, window.scrollX]);
    return ancestors;
}

function restoreScrollPositions(positions) {
    for (const [target, top, left] of positions) {
        if (target === window) {
            window.scrollTo(left, top);
        } else {
            target.scrollTop = top;
            target.scrollLeft = left;
        }
    }
}

function attachInlineTextareaEditor(box, saveValue) {
    const display = box.querySelector('.editable-display');
    const raw = box.querySelector('.raw-source');
    const status = box.querySelector('.save-status');
    if (!display || !raw || !status) return;

    display.addEventListener('click', () => {
        if (box.dataset.editing === '1') return;
        box.dataset.editing = '1';

        const textarea = document.createElement('textarea');
        textarea.className = 'edit-textarea';
        textarea.value = box.dataset.editAtTop ? `\n${raw.value}` : raw.value;
        textarea.style.height = `${Math.max(display.offsetHeight, box.offsetHeight - 30, 120)}px`;

        const scrollPositions = getScrollableAncestors(box);
        display.style.display = 'none';
        box.insertBefore(textarea, status);
        textarea.focus({ preventScroll: true });
        if (box.dataset.editAtTop) {
            requestAnimationFrame(() => {
                textarea.setSelectionRange(0, 0);
                textarea.scrollTop = 0;
                restoreScrollPositions(scrollPositions);
            });
        }

        let closed = false;
        const finish = async (save) => {
            if (closed) return;
            closed = true;
            const nextValue = textarea.value;
            textarea.remove();
            display.style.display = '';
            box.dataset.editing = '0';

            if (!save || nextValue === raw.value) return;

            status.textContent = 'Saving...';
            try {
                const savedValue = await saveValue(nextValue);
                raw.value = savedValue;
                renderMarkedText(display, savedValue.trim());
                status.textContent = 'Saved.';
            } catch (error) {
                status.textContent = String(error.message || 'Save failed.');
            }
        };

        textarea.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                finish(false);
            }
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                event.preventDefault();
                textarea.blur();
            }
        });
        textarea.addEventListener('blur', () => finish(true));
    });
}
