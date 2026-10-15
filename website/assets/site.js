const root = document.documentElement;
let savedTheme;
try { savedTheme = localStorage.getItem('seedl-theme'); } catch { /* Storage is optional. */ }
root.dataset.theme = savedTheme === 'light' ? 'light' : 'dark';
const theme = document.querySelector('[data-theme-toggle]');
function updateThemeLabel() {
  theme.textContent = root.dataset.theme === 'dark' ? '☀' : '☾';
  theme.setAttribute('aria-label', `Use ${root.dataset.theme === 'dark' ? 'light' : 'dark'} theme`);
}
updateThemeLabel();
theme.addEventListener('click', () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('seedl-theme', root.dataset.theme); } catch { /* Storage is optional. */ }
  updateThemeLabel();
});
const menu = document.querySelector('[data-menu]');
menu.addEventListener('click', () => {
  const expanded = menu.getAttribute('aria-expanded') === 'true';
  menu.setAttribute('aria-expanded', String(!expanded));
  document.querySelector('.navlinks').classList.toggle('open', !expanded);
});
document.querySelectorAll('.navlinks a').forEach(link => link.addEventListener('click', () => {
  menu.setAttribute('aria-expanded', 'false');
  document.querySelector('.navlinks').classList.remove('open');
}));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
    menu.click(); menu.focus();
  }
});
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(button.dataset.copy);
    button.textContent = 'Copied';
  } catch { button.textContent = 'Select command'; }
  setTimeout(() => { button.textContent = 'Copy'; }, 1800);
}));

const preview = document.querySelector('[data-preview]');
preview?.addEventListener('click', () => {
  const container = document.createElement('div');
  container.className = 'preview-live';
  const iframe = document.createElement('iframe');
  iframe.title = 'Interactive SeeDL 3D neural network preview';
  iframe.src = './demos/network.html';
  container.append(iframe);
  document.querySelector('.image-link').replaceWith(container);
  preview.disabled = true;
  preview.textContent = 'Drag to orbit · Click to inspect';
});
