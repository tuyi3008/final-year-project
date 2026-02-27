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
            <div class="empty-state">
                <i class="bi bi-images"></i>
                <h4>No images in gallery yet</h4>
                <p class="text-muted">Be the first to publish your creations!</p>
            </div>
        `;
        return;
    }
    
    galleryGrid.innerHTML = '';
    
    data.forEach(item => {
        // 不再创建 col 包装器，直接创建 gallery-item
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        
        // Format date
        const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Unknown date';
        
        // Check if user is logged in
        const isLoggedIn = !!localStorage.getItem('token');
        
        // Check if current user has liked this image (only if logged in)
        const isLiked = isLoggedIn ? checkIfUserLiked(item._id) : false;
        
        galleryItem.innerHTML = `
            <div class="gallery-image">
                <img src="/${item.image_path}" alt="${item.style} style">
                <div class="gallery-info">
                    <h6>${item.style.charAt(0).toUpperCase() + item.style.slice(1)} Style</h6>
                    <p class="small">
                        <i class="bi bi-person-circle"></i> ${item.username || 'Anonymous'}
                    </p>
                    <p class="small">
                        <i class="bi bi-calendar3"></i> ${date}
                    </p>
                    <div class="d-flex justify-content-between align-items-center mt-2">
                        <button class="btn-like ${isLiked ? 'liked' : ''}" 
                                onclick="likeGalleryItem('${item._id}', this)"
                                data-likes="${item.likes || 0}">
                            <i class="bi ${isLiked ? 'bi-heart-fill' : 'bi-heart'}"></i>
                            <span class="likes-count">${item.likes || 0}</span>
                        </button>
                        <button class="btn-download" onclick="downloadGalleryImage('${item.image_path}')">
                            <i class="bi bi-download"></i>
                            <span>Download</span>
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        galleryGrid.appendChild(galleryItem);
    });
    
    // Update stats
    updateStats();
}

// Function to check if user liked an image
function checkIfUserLiked(imageId) {
    const likedImages = JSON.parse(localStorage.getItem('likedGalleryImages') || '[]');
    return likedImages.includes(imageId);
}

// Function to like/unlike gallery item
async function likeGalleryItem(imageId, buttonElement) {
    // Check if user is logged in
    const token = localStorage.getItem('token');
    if (!token) {
        // Show login modal
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal('Please login to like images');
        } else {
            alert('Please login to like images');
        }
        return;
    }
    
    const isLiked = buttonElement.classList.contains('liked');
    const likesSpan = buttonElement.querySelector('.likes-count');
    let currentLikes = parseInt(likesSpan.textContent);
    
    // Disable button during API call
    buttonElement.disabled = true;
    
    try {
        const response = await fetch(`http://localhost:8000/api/gallery/${imageId}/like`, {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('Failed to update like');
        }
        
        const data = await response.json();
        
        // Update UI based on response
        if (data.liked) {
            // Liked
            buttonElement.classList.add('liked');
            buttonElement.querySelector('i').className = 'bi bi-heart-fill';
            likesSpan.textContent = currentLikes + 1;
            
            // Update localStorage
            const likedImages = JSON.parse(localStorage.getItem('likedGalleryImages') || '[]');
            likedImages.push(imageId);
            localStorage.setItem('likedGalleryImages', JSON.stringify(likedImages));
        } else {
            // Unliked
            buttonElement.classList.remove('liked');
            buttonElement.querySelector('i').className = 'bi bi-heart';
            likesSpan.textContent = currentLikes - 1;
            
            // Update localStorage
            const likedImages = JSON.parse(localStorage.getItem('likedGalleryImages') || '[]');
            const updatedLiked = likedImages.filter(id => id !== imageId);
            localStorage.setItem('likedGalleryImages', JSON.stringify(updatedLiked));
        }
        
    } catch (error) {
        console.error('Error updating like:', error);
        alert('Failed to update like. Please try again.');
    } finally {
        // Re-enable button
        buttonElement.disabled = false;
    }
}

// Function to download gallery image
function downloadGalleryImage(imagePath) {
    // Download the image
    const link = document.createElement('a');
    link.href = `/${imagePath}`;
    link.download = imagePath.split('/').pop();
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Show success message
    showDownloadMessage();
    
    // Track download in backend (optional, doesn't require login)
    const token = localStorage.getItem('token');
    if (token) {
        fetch(`http://localhost:8000/api/gallery/download`, {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_path: imagePath })
        }).catch(err => console.error('Error tracking download:', err));
    }
}

// Function to show download success message
function showDownloadMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    messageDiv.style.zIndex = '9999';
    messageDiv.innerHTML = `
        <i class="bi bi-check-circle-fill me-2"></i>
        Image downloaded successfully!
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
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
        
        const response = await fetch('http://localhost:8000/gallery/images');
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

// Make functions global for onclick handlers
window.likeGalleryItem = likeGalleryItem;
window.downloadGalleryImage = downloadGalleryImage;