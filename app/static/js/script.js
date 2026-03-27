document.addEventListener('DOMContentLoaded', function() {
    const dropdown = document.getElementById('comp-dropdown');
    if (dropdown) {
        dropdown.addEventListener('change', function() {
            const selectedComp = this.value.replace(/\//g, '---');
            window.location.href = `/refresh/${selectedComp}`;
        });
    }
});

function copyToClipboard() {
    const button = event.target;
    const parentDiv = button.closest('div');
    const siblings = Array.from(parentDiv.children).filter(el => el.tagName === 'P');
    const text = siblings.map(el => el.innerHTML).join('\n');
    navigator.clipboard.writeText(text).then(function() {
        const originalText = button.textContent;
        button.textContent = '[Copied!]';
        setTimeout(function() {
            button.textContent = originalText;
        }, 1500);
    }).catch(function(err) {
        console.error('Failed to copy: ', err);
        alert('Failed to copy to clipboard');
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const current = document.querySelector('.sidebar-item.current');
    if (current) {
        current.scrollIntoView({ behavior: 'auto', block: 'center' });
    }
    renderMarkedElements(document);
});

function loadResponseBox(element) {
    if (!element.open || element.hasAttribute('data-loaded')) return;
    const idd = element.getAttribute('id');
    fetch(`/modelinteraction/${idd}`)
        .then(response => response.text())
        .then(data => {
            const wrapper = document.createElement('div');
            wrapper.className = 'conversation-content';
            wrapper.innerHTML = data;
            element.appendChild(wrapper);
            element.setAttribute('data-loaded', true);
            renderMarkedElements(wrapper);
            if (window.hljs) hljs.highlightAll();
        })
        .catch(error => console.error('Error fetching details:', error));
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

    fetch(`/historystep/${stepId}`)
        .then(response => response.text())
        .then(data => {
            const element = document.getElementById(`history-step-content-${index}`);
            element.innerHTML = data;
            renderMarkedElements(element);
            if (window.hljs) hljs.highlightAll();
        })
        .catch(error => {
            console.error('Error fetching step:', error);
            document.getElementById(`history-step-content-${index}`).innerHTML = '<div class="error">Error loading step</div>';
        });
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.getElementById('sidebar-toggle-button');
    const isCollapsed = sidebar.classList.toggle('collapsed');
    if (toggleButton) toggleButton.innerHTML = isCollapsed ? '&#9776;' : '&times;';
}

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
