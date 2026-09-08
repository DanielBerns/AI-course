/**
 * AI Foundations Reference Library — Client Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let references = window.AI_REFERENCES || [];
  let currentCategory = 'All';
  let currentSearch = '';
  let currentSort = 'newest';
  let activeModalRef = null;

  // DOM Elements
  const gridContainer = document.getElementById('references-grid');
  const categoriesContainer = document.getElementById('categories-container');
  const searchInput = document.getElementById('search-input');
  const sortSelect = document.getElementById('sort-select');
  const themeToggleBtn = document.getElementById('theme-toggle');
  
  const modalOverlay = document.getElementById('modal-overlay');
  const modalPaperTitle = document.getElementById('modal-paper-title');
  const modalDetails = document.getElementById('modal-details');
  const modalBibtex = document.getElementById('modal-bibtex');
  const modalCloseBtn = document.getElementById('modal-close');
  const modalCopyBtn = document.getElementById('modal-copy-btn');
  const toast = document.getElementById('toast');

  const statTotal = document.getElementById('stat-total');
  const statPdf = document.getElementById('stat-pdf');
  const statCategories = document.getElementById('stat-categories');

  // Fallback fetch if window.AI_REFERENCES not loaded
  if (!references || references.length === 0) {
    fetch('data/references.json')
      .then(res => res.json())
      .then(data => {
        references = data;
        initApp();
      })
      .catch(err => console.error('Failed to load references:', err));
  } else {
    initApp();
  }

  function initApp() {
    setupTheme();
    updateStats();
    buildCategoryPills();
    renderGrid();
    setupEventListeners();
  }

  // Update counter stats
  function updateStats() {
    if (statTotal) statTotal.textContent = references.length;
    
    const pdfCount = references.filter(r => r.pdf && r.pdf.trim() !== '').length;
    if (statPdf) statPdf.textContent = pdfCount;

    const categories = new Set(references.map(r => r.category).filter(Boolean));
    if (statCategories) statCategories.textContent = categories.size;
  }

  // Build category filter pills
  function buildCategoryPills() {
    const categoryCounts = {};
    references.forEach(r => {
      const cat = r.category || 'General';
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    });

    const categories = ['All', ...Object.keys(categoryCounts).sort()];
    categoriesContainer.innerHTML = '';

    categories.forEach(cat => {
      const pill = document.createElement('button');
      pill.className = `category-pill ${cat === currentCategory ? 'active' : ''}`;
      const count = cat === 'All' ? references.length : categoryCounts[cat];
      pill.innerHTML = `${cat} <span style="opacity:0.75; font-size:0.8em; margin-left:4px;">(${count})</span>`;
      
      pill.addEventListener('click', () => {
        currentCategory = cat;
        document.querySelectorAll('.category-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderGrid();
      });

      categoriesContainer.appendChild(pill);
    });
  }

  // Sort and filter logic
  function getFilteredReferences() {
    return references.filter(ref => {
      const matchesCat = currentCategory === 'All' || ref.category === currentCategory;
      
      const q = currentSearch.toLowerCase();
      const matchesSearch = !q || (
        (ref.title && ref.title.toLowerCase().includes(q)) ||
        (ref.author && ref.author.toLowerCase().includes(q)) ||
        (ref.venue && ref.venue.toLowerCase().includes(q)) ||
        (ref.description && ref.description.toLowerCase().includes(q)) ||
        (ref.year && ref.year.toString().includes(q))
      );

      return matchesCat && matchesSearch;
    }).sort((a, b) => {
      if (currentSort === 'newest') return (b.year || 0) - (a.year || 0);
      if (currentSort === 'oldest') return (a.year || 0) - (b.year || 0);
      if (currentSort === 'title') return (a.title || '').localeCompare(b.title || '');
      if (currentSort === 'author') return (a.author || '').localeCompare(b.author || '');
      return 0;
    });
  }

  // Render Grid Cards
  function renderGrid() {
    const items = getFilteredReferences();
    gridContainer.innerHTML = '';

    if (items.length === 0) {
      gridContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon"><i class="fa-solid fa-folder-open"></i></div>
          <h3>No references found</h3>
          <p style="color: var(--text-secondary); margin-top: 6px;">Try adjusting your search query or filter category.</p>
        </div>
      `;
      return;
    }

    items.forEach(ref => {
      const card = document.createElement('article');
      card.className = 'ref-card';

      const pdfBtn = ref.pdf && ref.pdf.trim() !== '' 
        ? `<a href="${ref.pdf}" target="_blank" class="btn-action btn-pdf" title="Direct PDF Download"><i class="fa-solid fa-file-pdf"></i> PDF</a>`
        : '';

      const mainUrl = ref.url || ref.pdf || '#';

      card.innerHTML = `
        <div>
          <div class="ref-header">
            <span class="ref-category-tag">${ref.category || 'General'}</span>
            <span class="ref-year-tag">${ref.year || ''}</span>
          </div>
          <h3 class="ref-title">${escapeHtml(ref.title)}</h3>
          <div class="ref-authors"><i class="fa-solid fa-user-pen" style="font-size: 0.85em; margin-right: 4px;"></i> ${escapeHtml(ref.author)}</div>
          <div class="ref-venue">${escapeHtml(ref.venue)}</div>
          <p class="ref-description">${escapeHtml(ref.description)}</p>
        </div>
        <div class="ref-actions">
          <a href="${mainUrl}" target="_blank" class="btn-action" title="Open Source Link">
            <i class="fa-solid fa-arrow-up-right-from-square"></i> Open Link
          </a>
          ${pdfBtn}
          <button class="btn-action btn-bib modal-trigger" title="View BibTeX Citation">
            <i class="fa-solid fa-quote-right"></i> BibTeX
          </button>
        </div>
      `;

      // Modal trigger button click listener
      const bibBtn = card.querySelector('.modal-trigger');
      bibBtn.addEventListener('click', () => openModal(ref));

      gridContainer.appendChild(card);
    });
  }

  // Open Citation Modal
  function openModal(ref) {
    activeModalRef = ref;
    modalPaperTitle.textContent = ref.title || 'Citation Details';
    
    modalDetails.innerHTML = `
      <div style="font-size: 0.95rem; margin-bottom: 8px;"><strong>Authors:</strong> ${escapeHtml(ref.author)}</div>
      <div style="font-size: 0.95rem; margin-bottom: 8px;"><strong>Venue / Publisher:</strong> ${escapeHtml(ref.venue)} (${ref.year})</div>
      <div style="font-size: 0.95rem; margin-bottom: 8px;"><strong>Category:</strong> <span class="ref-category-tag">${ref.category}</span></div>
      ${ref.url ? `<div style="font-size: 0.9rem; margin-bottom: 8px; word-break: break-all;"><strong>URL:</strong> <a href="${ref.url}" target="_blank" style="color: var(--accent-cyan);">${ref.url}</a></div>` : ''}
      ${ref.description ? `<p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 12px; font-style: italic;">"${escapeHtml(ref.description)}"</p>` : ''}
    `;

    modalBibtex.textContent = ref.raw_bib || `@article{${ref.id},\n  author = {${ref.author}},\n  title = {${ref.title}},\n  year = {${ref.year}}\n}`;
    
    modalOverlay.classList.add('active');
  }

  function closeModal() {
    modalOverlay.classList.remove('active');
  }

  // Copy BibTeX to clipboard
  function copyBibtex() {
    if (!activeModalRef) return;
    const textToCopy = modalBibtex.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast('BibTeX copied to clipboard!');
    }).catch(err => {
      console.error('Copy failed:', err);
    });
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2500);
  }

  // Event Listeners
  function setupEventListeners() {
    searchInput.addEventListener('input', (e) => {
      currentSearch = e.target.value;
      renderGrid();
    });

    sortSelect.addEventListener('change', (e) => {
      currentSort = e.target.value;
      renderGrid();
    });

    modalCloseBtn.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) closeModal();
    });

    modalCopyBtn.addEventListener('click', copyBibtex);

    themeToggleBtn.addEventListener('click', toggleTheme);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modalOverlay.classList.contains('active')) {
        closeModal();
      }
    });
  }

  // Theme Management
  function setupTheme() {
    const savedTheme = localStorage.getItem('ai_sources_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('ai_sources_theme', newTheme);
    updateThemeIcon(newTheme);
  }

  function updateThemeIcon(theme) {
    const icon = themeToggleBtn.querySelector('i');
    if (theme === 'dark') {
      icon.className = 'fa-solid fa-sun';
      themeToggleBtn.title = 'Switch to Light Theme';
    } else {
      icon.className = 'fa-solid fa-moon';
      themeToggleBtn.title = 'Switch to Dark Theme';
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }
});
