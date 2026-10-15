const tabs = [...document.querySelectorAll('[data-tab]')];
const frames = [...document.querySelectorAll('iframe[data-demo]')];
const neuron = document.querySelector('#neuron');
function loadFrame(frame) {
  if (!frame.getAttribute('src')) frame.src = `./demos/${frame.dataset.demo}.html`;
}
function activate(name, updateHash = true) {
  if (!tabs.some(tab => tab.dataset.tab === name)) name = 'network';
  tabs.forEach(tab => {
    const active = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.dataset.tab).hidden = !active;
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
    if (next !== undefined) { event.preventDefault(); tabs[next].focus(); activate(tabs[next].dataset.tab); }
  });
});
const networkFrame = document.querySelector('#network iframe');
frames.forEach(frame => frame.addEventListener('load', () => {
  if (!frame.getAttribute('src')) return;
  const document = frame.contentDocument;
  if (!document?.querySelector('.plotly-graph-div')) {
    if (!frame.parentElement.querySelector('.fail-note')) {
      const message = window.document.createElement('p');
      message.className = 'fail-note'; message.textContent = 'The demo could not load. Reload this page or use the Python source below.';
      frame.after(message);
    }
  }
  if (frame === networkFrame && frame.contentWindow.__seedl_select_neuron) {
    const win = frame.contentWindow;
    const select = win.__seedl_select_neuron;
    const clear = win.__seedl_clear_selection;
    win.__seedl_select_neuron = id => { neuron.value = id; return select(id); };
    win.__seedl_clear_selection = () => { neuron.value = ''; return clear(); };
  }
}));
neuron.addEventListener('change', () => {
  const win = networkFrame.contentWindow;
  if (neuron.value) win?.__seedl_select_neuron?.(neuron.value);
  else win?.__seedl_clear_selection?.();
});
document.querySelector('#reset-network').addEventListener('click', () => {
  neuron.value = '';
  // Reload the already cached export to restore camera, selection, and legend.
  networkFrame.src = networkFrame.src;
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
