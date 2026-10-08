/**
 * AI Foundations Reference Library — Client Application Logic
 */

(function () {
  // Global State
  let currentCategory = 'All';
  let currentYear = 'All';
  let currentSearch = '';
  let currentSort = 'newest';
  let activeModalRef = null;

  function getReferences() {
    return window.AI_REFERENCES || [];
  }

  function initApp() {
    setupTheme();
    buildCategoryPills();
    buildYearPills();
    renderGrid();
    setupEventListeners();
  }

  // Build Research Area (Category) filter pills
  function buildCategoryPills() {
    const categoriesContainer = document.getElementById('categories-container');
    if (!categoriesContainer) return;

    const refs = getReferences();
    const categoryCounts = {};
    refs.forEach(r => {
      const cat = r.category || 'General';
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    });

    const categories = ['All', ...Object.keys(categoryCounts).sort()];
    categoriesContainer.innerHTML = '';

    categories.forEach(cat => {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = `category-pill ${cat === currentCategory ? 'active' : ''}`;
      const count = cat === 'All' ? refs.length : categoryCounts[cat];
      const label = cat === 'All' ? 'All Areas' : cat;
      pill.innerHTML = `${label} <span class="pill-badge">(${count})</span>`;
      
      pill.addEventListener('click', () => {
        currentCategory = cat;
        categoriesContainer.querySelectorAll('.category-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderGrid();
      });

      categoriesContainer.appendChild(pill);
    });
  }

  // Define Year Filter Options
  const YEAR_OPTIONS = [
    { key: 'All', label: 'All Years' },
    { key: '2024', label: '2024' },
    { key: '2023', label: '2023' },
    { key: '2022', label: '2022' },
    { key: '2021', label: '2021' },
    { key: '2020', label: '2020' },
    { key: '2019', label: '2019' },
    { key: '2018', label: '2018' },
    { key: '2017', label: '2017' },
    { key: '2016', label: '2016' },
    { key: '2015', label: '2015' },
    { key: '2010–2014', label: '2010–2014' },
    { key: '2000s', label: '2000–2009' },
    { key: '1980–1990s', label: '1980–1999' },
    { key: 'Pre-1980', label: 'Pre-1980' }
  ];

  // Helper: Test if paper year matches filter key
  function matchesYear(yearStr, filterKey) {
    if (filterKey === 'All') return true;
    if (!yearStr) return false;
    const y = parseInt(yearStr, 10);
    if (!y || isNaN(y)) return false;

    if (filterKey === '2010–2014' || filterKey === '2010-2014') return y >= 2010 && y <= 2014;
    if (filterKey === '2000s' || filterKey === '2000-2009') return y >= 2000 && y <= 2009;
    if (filterKey === '1980–1990s' || filterKey === '1980-1999') return y >= 1980 && y <= 1999;
    if (filterKey === 'Pre-1980') return y < 1980;

    return y.toString() === filterKey;
  }

  // Build Year / Era filter pills
  function buildYearPills() {
    const yearsContainer = document.getElementById('years-container');
    if (!yearsContainer) return;

    const refs = getReferences();
    yearsContainer.innerHTML = '';

    YEAR_OPTIONS.forEach(opt => {
      const count = opt.key === 'All' 
        ? refs.length 
        : refs.filter(r => matchesYear(r.year, opt.key)).length;

      // Only render pills that have matching papers or are 'All'
      if (count === 0 && opt.key !== 'All') return;

      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = `category-pill year-pill ${opt.key === currentYear ? 'active' : ''}`;
      pill.innerHTML = `${opt.label} <span class="pill-badge">(${count})</span>`;

      pill.addEventListener('click', () => {
        currentYear = opt.key;
        yearsContainer.querySelectorAll('.year-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderGrid();
      });

      yearsContainer.appendChild(pill);
    });
  }

  // Filter and sort logic
  function getFilteredReferences() {
    const refs = getReferences();
    return refs.filter(ref => {
      const matchesCat = currentCategory === 'All' || ref.category === currentCategory;
      const matchesYr = matchesYear(ref.year, currentYear);
      
      const q = currentSearch.toLowerCase();
      const matchesSearch = !q || (
        (ref.title && ref.title.toLowerCase().includes(q)) ||
        (ref.author && ref.author.toLowerCase().includes(q)) ||
        (ref.venue && ref.venue.toLowerCase().includes(q)) ||
        (ref.description && ref.description.toLowerCase().includes(q)) ||
        (ref.year && ref.year.toString().includes(q))
      );

      return matchesCat && matchesYr && matchesSearch;
    }).sort((a, b) => {
      if (currentSort === 'newest') return (parseInt(b.year) || 0) - (parseInt(a.year) || 0);
      if (currentSort === 'oldest') return (parseInt(a.year) || 0) - (parseInt(b.year) || 0);
      if (currentSort === 'title') return (a.title || '').localeCompare(b.title || '');
      if (currentSort === 'author') return (a.author || '').localeCompare(b.author || '');
      return 0;
    });
  }

  // Render Grid Cards and Update Status Summary
  function renderGrid() {
    const gridContainer = document.getElementById('references-grid');
    const resultsCount = document.getElementById('results-count');
    const resetFiltersBtn = document.getElementById('reset-filters-btn');
    if (!gridContainer) return;

    const refs = getReferences();
    const items = getFilteredReferences();
    gridContainer.innerHTML = '';

    // Update Results Bar
    const hasActiveFilters = currentCategory !== 'All' || currentYear !== 'All' || currentSearch.trim() !== '';
    if (resetFiltersBtn) {
      resetFiltersBtn.style.display = hasActiveFilters ? 'inline-flex' : 'none';
    }

    if (resultsCount) {
      let text = `Showing <strong>${items.length}</strong> of ${refs.length} papers`;
      if (currentCategory !== 'All') text += ` • Area: <em>${escapeHtml(currentCategory)}</em>`;
      if (currentYear !== 'All') text += ` • Year: <em>${escapeHtml(currentYear)}</em>`;
      if (currentSearch.trim()) text += ` • Search: "<em>${escapeHtml(currentSearch)}</em>"`;
      resultsCount.innerHTML = text;
    }

    if (items.length === 0) {
      gridContainer.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon"><i class="fa-solid fa-folder-open"></i></div>
          <h3>No matching papers found</h3>
          <p style="color: var(--text-secondary); margin-top: 6px;">Try clearing filters or adjusting your search term.</p>
          <button type="button" class="btn-primary" style="margin-top: 16px;" onclick="window.resetFiltersTrigger()">
            <i class="fa-solid fa-rotate-left"></i> Reset All Filters
          </button>
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
          <button type="button" class="btn-action btn-bib modal-trigger" title="View BibTeX Citation">
            <i class="fa-solid fa-quote-right"></i> BibTeX
          </button>
        </div>
      `;

      // Modal trigger button click listener
      const bibBtn = card.querySelector('.modal-trigger');
      if (bibBtn) {
        bibBtn.addEventListener('click', () => openModal(ref));
      }

      gridContainer.appendChild(card);
    });
  }

  // Reset all active filters
  function resetAllFilters() {
    currentCategory = 'All';
    currentYear = 'All';
    currentSearch = '';
    const searchInput = document.getElementById('search-input');
    if (searchInput) searchInput.value = '';

    buildCategoryPills();
    buildYearPills();
    renderGrid();
  }
  window.resetFiltersTrigger = resetAllFilters;

  // Open Citation Modal
  function openModal(ref) {
    const modalOverlay = document.getElementById('modal-overlay');
    const modalPaperTitle = document.getElementById('modal-paper-title');
    const modalDetails = document.getElementById('modal-details');
    const modalBibtex = document.getElementById('modal-bibtex');
    if (!modalOverlay) return;

    activeModalRef = ref;
    if (modalPaperTitle) modalPaperTitle.textContent = ref.title || 'Citation Details';
    
    if (modalDetails) {
      modalDetails.innerHTML = `
        <div style="font-size: 0.95rem; margin-bottom: 8px;"><strong>Authors:</strong> ${escapeHtml(ref.author)}</div>
        <div style="font-size: 0.95rem; margin-bottom: 8px;"><strong>Venue / Publisher:</strong> ${escapeHtml(ref.venue)} (${ref.year})</div>
        <div style="font-size: 0.95rem; margin-bottom: 8px;"><strong>Category:</strong> <span class="ref-category-tag">${ref.category}</span></div>
        ${ref.url ? `<div style="font-size: 0.9rem; margin-bottom: 8px; word-break: break-all;"><strong>URL:</strong> <a href="${ref.url}" target="_blank" style="color: var(--accent-cyan);">${ref.url}</a></div>` : ''}
        ${ref.description ? `<p style="font-size: 0.9rem; color: var(--text-secondary); margin-top: 12px; font-style: italic;">"${escapeHtml(ref.description)}"</p>` : ''}
      `;
    }

    if (modalBibtex) {
      modalBibtex.textContent = ref.raw_bib || `@article{${ref.id},\n  author = {${ref.author}},\n  title = {${ref.title}},\n  year = {${ref.year}}\n}`;
    }
    
    modalOverlay.classList.add('active');
  }

  function closeModal() {
    const modalOverlay = document.getElementById('modal-overlay');
    if (modalOverlay) modalOverlay.classList.remove('active');
  }

  // Copy BibTeX to clipboard
  function copyBibtex() {
    if (!activeModalRef) return;
    const modalBibtex = document.getElementById('modal-bibtex');
    if (!modalBibtex) return;
    const textToCopy = modalBibtex.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast('BibTeX copied to clipboard!');
    }).catch(err => {
      console.error('Copy failed:', err);
    });
  }

  function showToast(msg) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2500);
  }

  // Event Listeners
  function setupEventListeners() {
    const searchInput = document.getElementById('search-input');
    const sortSelect = document.getElementById('sort-select');
    const resetFiltersBtn = document.getElementById('reset-filters-btn');
    const modalCloseBtn = document.getElementById('modal-close');
    const modalOverlay = document.getElementById('modal-overlay');
    const modalCopyBtn = document.getElementById('modal-copy-btn');
    const themeToggleBtn = document.getElementById('theme-toggle');

    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        currentSearch = e.target.value;
        renderGrid();
      });
    }

    if (sortSelect) {
      sortSelect.addEventListener('change', (e) => {
        currentSort = e.target.value;
        renderGrid();
      });
    }

    if (resetFiltersBtn) {
      resetFiltersBtn.addEventListener('click', resetAllFilters);
    }

    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    if (modalOverlay) {
      modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
      });
    }

    if (modalCopyBtn) modalCopyBtn.addEventListener('click', copyBibtex);
    if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modalOverlay && modalOverlay.classList.contains('active')) {
        closeModal();
      }
    });
  }

  // Theme Management
  function setupTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('ai_sources_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    if (themeToggleBtn) updateThemeIcon(savedTheme);
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('ai_sources_theme', newTheme);
    updateThemeIcon(newTheme);
  }

  function updateThemeIcon(theme) {
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (!themeToggleBtn) return;
    const icon = themeToggleBtn.querySelector('i');
    if (!icon) return;
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

  // Execute initialization cleanly across all loading states
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();
