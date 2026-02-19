// gallery.js - Gallery with Filter Functionality

let currentFilter = 'all';
let galleryData = []; // Will be populated from backend

// Function to filter gallery items
function filterGallery(filter) {
    currentFilter = filter;
    const filteredData = galleryData.filter(item => {
        if (filter === 'all') return true;
        return item.style === filter;
    });
    renderGallery(filteredData);
}

// Function to render gallery
function renderGallery(data = galleryData) {
    const galleryGrid = document.getElementById('gallery-grid');
    
    if (!galleryGrid) return;
    
    if (data.length === 0) {
        galleryGrid.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="bi bi-images display-1 text-muted mb-3"></i>
                <h4 class="text-muted">No images in gallery yet</h4>
                <p class="text-muted">Be the first to publish your creations!</p>
            </div>
        `;
        return;
    }
    
    galleryGrid.innerHTML = '';
    
    data.forEach(item => {
        const col = document.createElement('div');
        col.className = 'col-md-4 col-lg-3';
        
        // Format date
        const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Unknown date';
        
        col.innerHTML = `
            <div class="gallery-item">
                <img src="/${item.image_path}" alt="${item.style} style">
                <div class="gallery-info">
                    <h6 class="mb-1">${item.style.charAt(0).toUpperCase() + item.style.slice(1)} Style</h6>
                    <p class="small text-muted mb-1">
                        <i class="bi bi-person-circle me-1"></i>${item.username || 'Anonymous'}
                    </p>
                    <p class="small text-muted mb-2">
                        <i class="bi bi-calendar3 me-1"></i>${date}
                    </p>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="badge bg-primary">${item.likes || 0} ❤️</span>
                        <span class="badge bg-secondary">${item.views || 0} 👁️</span>
                    </div>
                </div>
            </div>
        `;
        
        galleryGrid.appendChild(col);
    });
    
    // Update stats
    updateStats();
}

// Function to update stats (total images per style)
function updateStats() {
    const totalCount = galleryData.length;
    const sketchCount = galleryData.filter(item => item.style === 'sketch').length;
    const animeCount = galleryData.filter(item => item.style === 'anime').length;
    const inkCount = galleryData.filter(item => item.style === 'ink').length;
    
    // Update counters if they exist in DOM
    const totalEl = document.getElementById('total-count');
    if (totalEl) totalEl.textContent = totalCount;
    
    const sketchEl = document.getElementById('sketch-count');
    if (sketchEl) sketchEl.textContent = sketchCount;
    
    const animeEl = document.getElementById('anime-count');
    if (animeEl) animeEl.textContent = animeCount;
    
    const inkEl = document.getElementById('ink-count');
    if (inkEl) inkEl.textContent = inkCount;
}

// Function to load gallery images from backend
async function loadGalleryFromBackend() {
    const galleryGrid = document.getElementById('gallery-grid');
    
    try {
        // Show loading state
        galleryGrid.innerHTML = `
            <div class="col-12 text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3 text-muted">Loading gallery...</p>
            </div>
        `;
        
        const response = await fetch('/gallery/images');
        const data = await response.json();
        
        if (data.code === 200 && data.images && data.images.length > 0) {
            galleryData = data.images;
            renderGallery();
        } else {
            galleryGrid.innerHTML = `
                <div class="col-12 text-center py-5">
                    <i class="bi bi-images display-1 text-muted mb-3"></i>
                    <h4 class="text-muted">No images in gallery yet</h4>
                    <p class="text-muted">Be the first to publish your creations!</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading gallery:', error);
        galleryGrid.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="bi bi-exclamation-triangle display-1 text-muted mb-3"></i>
                <h4 class="text-muted">Failed to load gallery</h4>
                <p class="text-muted">Please try again later</p>
            </div>
        `;
    }
}

// Setup filter button events
function setupFilterButtons() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Filter gallery
            const filter = button.getAttribute('data-filter');
            filterGallery(filter);
        });
    });
}

// Initialize gallery when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadGalleryFromBackend();
    setupFilterButtons();
});