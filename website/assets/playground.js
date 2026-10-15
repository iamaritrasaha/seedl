const tabs = [...document.querySelectorAll('[data-tab]')];
const frames = [...document.querySelectorAll('iframe[data-demo]')];
const neuron = document.querySelector('#neuron');

function loadFrame(frame) {
  if (frame && !frame.getAttribute('src')) frame.src = `./demos/${frame.dataset.demo}.html`;
}

function activate(name, updateHash = true) {
  if (!tabs.some(tab => tab.dataset.tab === name)) name = tabs[0]?.dataset.tab || 'cnn';
  tabs.forEach(tab => {
    const active = tab.dataset.tab === name;
    if (!active) {
      document.querySelector(`#${tab.dataset.tab} iframe`)?.contentDocument
        ?.querySelectorAll('.plotly-graph-div').forEach(plot => plot.seedlPlayback?.pause());
    }
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(tab.dataset.tab);
    if (panel) panel.hidden = !active;
  });
  const frame = document.querySelector(`#${name} iframe`);
  loadFrame(frame);
  if (updateHash) history.replaceState(null, '', `#${name}`);
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activate(tab.dataset.tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      tabs[next].focus();
      activate(tabs[next].dataset.tab);
    }
  });
});

const networkFrame = document.querySelector('#network iframe');
frames.forEach(frame => frame.addEventListener('load', () => {
  if (!frame.getAttribute('src')) return;
  const doc = frame.contentDocument;
  const isLoaded = doc && (
    doc.querySelector('.plotly-graph-div') ||
    doc.querySelector('canvas') ||
    doc.querySelector('svg') ||
    doc.querySelector('.token-table') ||
    doc.querySelector('.container') ||
    doc.querySelector('.features-container')
  );
  if (!isLoaded) {
    if (!frame.parentElement.querySelector('.fail-note')) {
      const message = window.document.createElement('p');
      message.className = 'fail-note';
      message.textContent = 'The demo could not load. Reload this page or use the Python source below.';
      frame.after(message);
    }
  }
  if (frame === networkFrame && frame.contentWindow && frame.contentWindow.__seedl_select_neuron) {
    const win = frame.contentWindow;
    const select = win.__seedl_select_neuron;
    const clear = win.__seedl_clear_selection;
    win.__seedl_select_neuron = id => { if (neuron) neuron.value = id; return select(id); };
    win.__seedl_clear_selection = () => { if (neuron) neuron.value = ''; return clear(); };
  }
}));

if (neuron) {
  neuron.addEventListener('change', () => {
    const win = networkFrame?.contentWindow;
    if (neuron.value) win?.__seedl_select_neuron?.(neuron.value);
    else win?.__seedl_clear_selection?.();
  });
}

const resetBtn = document.querySelector('#reset-network');
if (resetBtn) {
  resetBtn.addEventListener('click', () => {
    if (neuron) neuron.value = '';
    if (networkFrame) networkFrame.src = networkFrame.src;
  });
}

const evolutionSelect = document.querySelector('#evolution-circuit');
if (evolutionSelect) {
  evolutionSelect.addEventListener('change', event => {
    const frame = document.querySelector('#evolution iframe');
    if (frame) {
      frame.contentDocument?.querySelectorAll('.plotly-graph-div')
        .forEach(plot => plot.seedlPlayback?.pause());
      frame.dataset.demo = event.target.value;
      frame.src = `./demos/${frame.dataset.demo}.html`;
    }
  });
}

const attentionPairSelect = document.querySelector('#attention-pair');
if (attentionPairSelect) {
  attentionPairSelect.addEventListener('change', event => {
    const frame = document.querySelector('#attention-cmp iframe');
    if (frame) {
      frame.dataset.demo = event.target.value;
      frame.src = `./demos/${frame.dataset.demo}.html`;
    }
  });
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    frames.forEach(frame => frame.contentDocument
      ?.querySelectorAll('.plotly-graph-div').forEach(plot => plot.seedlPlayback?.pause()));
  }
});

let manifestPromise;
function getManifest() {
  if (!manifestPromise) manifestPromise = fetch('./demos/manifest.json').then(response => {
    if (!response.ok) throw new Error('Demo manifest unavailable');
    return response.json();
  }).catch(error => { manifestPromise = null; throw error; });
  return manifestPromise;
}

let stateRequest = 0;
document.querySelectorAll('[data-state]').forEach(button => button.addEventListener('click', async () => {
  const request = ++stateRequest;
  try {
    const manifest = await getManifest();
    if (request !== stateRequest) return;
    const data = manifest.bloch[button.dataset.state];
    document.querySelectorAll('[data-state]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    document.querySelector('#bloch-vector').textContent = `(${data.vector.map(n => Math.abs(n) < 1e-12 ? 0 : Math.round(n)).join(', ')})`;
    document.querySelector('#bloch-prob').textContent = `P(0) = ${Math.round(data.prob_0 * 100)}% · P(1) = ${Math.round(data.prob_1 * 100)}%`;
    document.querySelector('#bloch iframe').src = `./demos/bloch-${button.dataset.state}.html`;
  } catch {
    document.querySelector('#bloch-vector').textContent = 'Unable to load preset data. Please reload.';
  }
}));

window.addEventListener('hashchange', () => activate(location.hash.slice(1), false));
activate(location.hash.slice(1), false);
