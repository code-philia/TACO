/**
 * Repo2Doc Agent Demo - frontend interaction script.
 */

// DOM elements.
const repoPathInput = document.getElementById('repo-path');
const runBtn = document.getElementById('run-btn');
const logContainer = document.getElementById('log-container');
const docPreview = document.getElementById('doc-preview');
const clearLogBtn = document.getElementById('clear-log');
const missingSection = document.getElementById('missing-section');
const missingList = document.getElementById('missing-list');

// Summary elements.
const statIteration = document.getElementById('stat-iteration');
const statConfidence = document.getElementById('stat-confidence');
const statDocLength = document.getElementById('stat-doc-length');
const statStatus = document.getElementById('stat-status');

// Workflow node elements.
const workflowNodes = document.querySelectorAll('.workflow-node');

// Runtime state.
let eventSource = null;
let isRunning = false;

/**
 * Format a timestamp.
 */
function formatTime(date) {
    return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * Append a log entry.
 */
function addLog(type, message) {
    // Remove the placeholder node.
    const placeholder = logContainer.querySelector('.log-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `
        <span class="log-time">${formatTime(new Date())}</span>
        <span class="log-type ${type}">[${type.toUpperCase()}]</span>
        <span class="log-message">${escapeHtml(message)}</span>
    `;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

/**
 * Escape HTML.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Update the workflow node state.
 */
function updateNode(nodeName, status) {
    workflowNodes.forEach(node => {
        if (node.dataset.node === nodeName) {
            node.classList.remove('active', 'completed', 'error');
            if (status) {
                node.classList.add(status);
            }
            const statusEl = node.querySelector('.node-status');
            if (statusEl) {
                if (status === 'active') {
                    statusEl.textContent = 'Running...';
                } else if (status === 'completed') {
                    statusEl.textContent = 'Done ✓';
                } else if (status === 'error') {
                    statusEl.textContent = 'Error';
                } else {
                    statusEl.textContent = '';
                }
            }
        }
    });
}

/**
 * Reset all node states.
 */
function resetNodes() {
    workflowNodes.forEach(node => {
        node.classList.remove('active', 'completed', 'error');
        const statusEl = node.querySelector('.node-status');
        if (statusEl) {
            statusEl.textContent = '';
        }
    });
}

/**
 * Update the summary stats.
 */
function updateStats(data) {
    if (data.iteration !== undefined) {
        statIteration.textContent = data.iteration;
    }
    if (data.confidence !== undefined) {
        statConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    }
    if (data.document_length !== undefined) {
        statDocLength.textContent = data.document_length.toLocaleString();
    }
}

/**
 * Set the running state.
 */
function setRunningState(running) {
    isRunning = running;
    runBtn.disabled = running;
    repoPathInput.disabled = running;

    if (running) {
        runBtn.classList.add('running');
        runBtn.querySelector('.btn-text').textContent = 'Running...';
        statStatus.textContent = 'Running';
        statStatus.className = 'stat-value status-badge running';
    } else {
        runBtn.classList.remove('running');
        runBtn.querySelector('.btn-text').textContent = 'Run Agent';
    }
}

/**
 * Set the completion state.
 */
function setCompletedState(success) {
    setRunningState(false);
    if (success) {
        statStatus.textContent = 'Completed';
        statStatus.className = 'stat-value status-badge completed';
    } else {
        statStatus.textContent = 'Failed';
        statStatus.className = 'stat-value status-badge error';
    }
}

/**
 * Handle SSE events.
 */
function handleSSEEvent(event, data) {
    switch (event) {
        case 'start':
            addLog('info', `Started repository analysis: ${data.repo_path}`);
            break;

        case 'node_start':
            updateNode(data.node, 'active');
            addLog('info', `Node [${data.node}] started (iteration ${data.iteration})`);
            break;

        case 'node_complete':
            updateNode(data.node, 'completed');
            addLog('success', `Node [${data.node}] completed`);
            updateStats({
                iteration: data.iteration,
                confidence: data.confidence,
                document_length: data.document_length
            });

            // Update the missing-parts panel.
            if (data.missing_parts && data.missing_parts.length > 0) {
                missingSection.style.display = 'block';
                missingList.innerHTML = data.missing_parts
                    .map(part => `<li>${escapeHtml(part)}</li>`)
                    .join('');
            }
            break;

        case 'exploration':
            if (data.tool_calls && data.tool_calls.length > 0) {
                const toolNames = data.tool_calls.map(tc => tc.tool_name).join(', ');
                addLog('info', `Tool calls: ${toolNames}`);
            }
            if (data.findings) {
                addLog('info', `Finding: ${data.findings.substring(0, 100)}...`);
            }
            break;

        case 'complete':
            addLog('success', `Analysis completed after ${data.iteration_count} iterations. Confidence ${(data.confidence_score * 100).toFixed(1)}%`);
            updateStats({
                iteration: data.iteration_count,
                confidence: data.confidence_score,
                document_length: data.document_length
            });

            // Show the document preview.
            if (data.document_preview) {
                docPreview.innerHTML = '';
                docPreview.textContent = data.document_preview;
                if (data.document_length > 2000) {
                    docPreview.textContent += '\n\n... (Document truncated; see the output file for the full version)';
                }
            }
            setCompletedState(true);
            missingSection.style.display = 'none';
            break;

        case 'failed':
            addLog('error', `Analysis failed: ${data.error}`);
            setCompletedState(false);
            break;

        case 'error':
            addLog('error', `Error: ${data.message || data.node}`);
            if (data.node) {
                updateNode(data.node, 'error');
            }
            break;

        case 'end':
            addLog('info', 'Task ended');
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            break;

        case 'heartbeat':
            // Ignore heartbeat events.
            break;

        default:
            console.log('Unknown event:', event, data);
    }
}

/**
 * Start an agent task.
 */
async function runAgent() {
    const repoPath = repoPathInput.value.trim();
    if (!repoPath) {
        alert('Please enter YOUR_REPO_PATH_HERE');
        return;
    }

    // Reset the UI.
    resetNodes();
    logContainer.innerHTML = '<div class="log-placeholder">Waiting to run...</div>';
    docPreview.innerHTML = '<div class="doc-placeholder">The document preview will appear here...</div>';
    missingSection.style.display = 'none';
    statIteration.textContent = '0';
    statConfidence.textContent = '0%';
    statDocLength.textContent = '0';

    setRunningState(true);
    addLog('info', 'Starting agent...');

    try {
        // Send the run request.
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ repo_path: repoPath })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Failed to start');
        }

        const taskId = result.task_id;
        addLog('success', `Task created: ${taskId}`);

        // Connect to the SSE event stream.
        eventSource = new EventSource(`/api/stream/${taskId}`);

        eventSource.onopen = () => {
            addLog('info', 'Connected to event stream');
        };

        eventSource.onerror = (e) => {
            console.error('SSE Error:', e);
            if (eventSource.readyState === EventSource.CLOSED) {
                addLog('warning', 'Event stream closed');
                setRunningState(false);
            }
        };

        // Subscribe to event types.
        const eventTypes = ['start', 'node_start', 'node_complete', 'exploration',
            'complete', 'failed', 'error', 'end', 'heartbeat'];

        eventTypes.forEach(eventType => {
            eventSource.addEventListener(eventType, (e) => {
                try {
                    const data = JSON.parse(e.data);
                    handleSSEEvent(eventType, data);
                } catch (err) {
                    console.error('Failed to parse event data:', err);
                }
            });
        });

    } catch (error) {
        addLog('error', `Failed to start: ${error.message}`);
        setCompletedState(false);
    }
}

// Event bindings.
runBtn.addEventListener('click', runAgent);

repoPathInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isRunning) {
        runAgent();
    }
});

clearLogBtn.addEventListener('click', () => {
    logContainer.innerHTML = '<div class="log-placeholder">Waiting to run...</div>';
});

// Clean up on page unload.
window.addEventListener('beforeunload', () => {
    if (eventSource) {
        eventSource.close();
    }
});
