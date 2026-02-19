// profile.js

document.addEventListener('DOMContentLoaded', function() {
    console.log('Profile page loaded');
    
    // Check if user is logged in
    if (!auth.isLoggedIn()) {
        window.location.href = '/';
        return;
    }
    
    // Load user profile data
    loadUserProfile();
    
    // Load user stats
    loadUserStats();
    
    // Load history
    loadHistory();
    
    // Load favorites
    loadFavorites();
});

function loadUserProfile() {
    const userEmail = localStorage.getItem('userEmail');
    document.getElementById('profileName').textContent = userEmail.split('@')[0] || 'User';
    document.getElementById('profileEmail').textContent = userEmail;
}

function loadUserStats() {
    // TODO: Fetch from backend
    document.getElementById('transformCount').textContent = '12';
    document.getElementById('favoriteCount').textContent = '5';
    document.getElementById('shareCount').textContent = '3';
}

async function loadHistory() {
    console.log('Loading history...');
    
    const historyContainer = document.getElementById('historyContainer');
    const token = localStorage.getItem('token');
    
    if (!token) return;
    
    try {
        historyContainer.innerHTML = `
            <div class="col-12 text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3 text-muted">Loading your history...</p>
            </div>
        `;
        
        const response = await fetch('/history', {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        
        const data = await response.json();
        
        if (data.code === 200 && data.history && data.history.length > 0) {
            // Check favorite status for each item
            const historyWithFav = await Promise.all(
                data.history.map(async (item) => {
                    const isFav = await checkFavoriteStatus(item.result_image_path);
                    return { ...item, isFavorite: isFav };
                })
            );
            displayHistory(historyWithFav);
        } else {
            historyContainer.innerHTML = `
                <div class="col-12 text-center py-5 text-muted">
                    <i class="bi bi-clock-history display-4 mb-3"></i>
                    <p>No transformation history yet</p>
                    <a href="/" class="btn btn-gradient mt-3">
                        <i class="bi bi-magic me-2"></i>Create Your First
                    </a>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

async function loadFavorites() {
    console.log('Loading favorites...');
    
    const favoritesContainer = document.getElementById('favoritesContainer');
    const token = localStorage.getItem('token');
    
    if (!token) return;
    
    try {
        const response = await fetch('/favorites', {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        
        const data = await response.json();
        
        if (data.code === 200 && data.favorites && data.favorites.length > 0) {
            displayFavorites(data.favorites);
        } else {
            favoritesContainer.innerHTML = `
                <div class="col-12 text-center py-5 text-muted">
                    <i class="bi bi-heart display-4 mb-3"></i>
                    <p>No favorites yet</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading favorites:', error);
    }
}

async function checkFavoriteStatus(imagePath) {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`/favorites/check?image_path=${encodeURIComponent(imagePath)}`, {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        const data = await response.json();
        return data.is_favorite || false;
    } catch (error) {
        return false;
    }
}

function displayHistory(history) {
    const historyContainer = document.getElementById('historyContainer');
    historyContainer.innerHTML = '';
    
    history.forEach(item => {
        const date = new Date(item.created_at);
        const formattedDate = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        const imageUrl = item.result_image_path ? `/${item.result_image_path}` : 'https://via.placeholder.com/80';
        const isFav = item.isFavorite || false;
        
        historyContainer.innerHTML += `
            <div class="col-md-4">
                <div class="history-item d-flex align-items-center gap-3">
                    <img src="${imageUrl}" alt="${item.style}" class="history-image">
                    <div>
                        <h6 class="mb-1">${item.style.charAt(0).toUpperCase() + item.style.slice(1)} Style</h6>
                        <p class="text-muted small mb-0">${formattedDate}</p>
                        <div class="mt-2 d-flex gap-2">
                            <!-- favorite button -->
                            <button class="btn-favorite ${isFav ? 'active' : ''}" 
                                    onclick="toggleFavorite('${item.result_image_path}', '${item.style}', this)">
                                <i class="bi ${isFav ? 'bi-heart-fill' : 'bi-heart'}"></i>
                            </button>
                            <!-- download button -->
                            <button class="btn btn-sm btn-outline-light" onclick="downloadImage('${item.result_image_path}')">
                                <i class="bi bi-download"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
}

function displayFavorites(favorites) {
    const favoritesContainer = document.getElementById('favoritesContainer');
    favoritesContainer.innerHTML = '';
    
    favorites.forEach(item => {
        const date = new Date(item.created_at);
        const formattedDate = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        const imageUrl = item.image_path ? `/${item.image_path}` : 'https://via.placeholder.com/80';
        
        favoritesContainer.innerHTML += `
            <div class="col-md-4">
                <div class="history-item d-flex align-items-center gap-3">
                    <img src="${imageUrl}" alt="${item.style}" class="history-image">
                    <div>
                        <h6 class="mb-1">${item.style.charAt(0).toUpperCase() + item.style.slice(1)} Style</h6>
                        <p class="text-muted small mb-0">${formattedDate}</p>
                        <div class="mt-2 d-flex gap-2">
                            <button class="btn-favorite active" 
                                    onclick="toggleFavorite('${item.image_path}', '${item.style}', this)">
                                <i class="bi bi-heart-fill"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-light" onclick="downloadImage('${item.image_path}')">
                                <i class="bi bi-download"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
}

async function toggleFavorite(imagePath, style, button) {
    const token = localStorage.getItem('token');
    const isFavorite = button.classList.contains('active');
    const icon = button.querySelector('i');
    
    try {
        const response = await fetch(isFavorite ? '/favorites/remove' : '/favorites/add', {
            method: isFavorite ? 'DELETE' : 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                image_path: imagePath,
                style: style 
            })
        });
        
        const data = await response.json();
        
        if (data.code === 200) {

            button.classList.toggle('active');
            icon.className = isFavorite ? 'bi bi-heart' : 'bi bi-heart-fill';
            
            const favCount = document.getElementById('favoriteCount');
            favCount.textContent = parseInt(favCount.textContent) + (isFavorite ? -1 : 1);
            
            const favoritesTab = document.getElementById('favorites-tab');
            if (favoritesTab && favoritesTab.classList.contains('active')) {
                console.log('Refreshing favorites tab...');
                await loadFavorites();
            }
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to update favorite');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('Profile page loaded');
    
    if (!auth.isLoggedIn()) {
        window.location.href = '/';
        return;
    }
    
    loadUserProfile();
    loadUserStats();
    loadHistory();
    loadFavorites();

    const favoritesTab = document.getElementById('favorites-tab');
    if (favoritesTab) {
        favoritesTab.addEventListener('shown.bs.tab', function() {
            console.log('Favorites tab shown, reloading...');
            loadFavorites();
        });
    }
    
    const historyTab = document.getElementById('history-tab');
    if (historyTab) {
        historyTab.addEventListener('shown.bs.tab', function() {
            console.log('History tab shown, reloading...');
            loadHistory();
        });
    }
});

function downloadImage(imagePath) {
    const link = document.createElement('a');
    link.href = `/${imagePath}`;
    link.download = imagePath.split('/').pop();
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Make functions global for onclick handlers
window.toggleFavorite = toggleFavorite;
window.downloadImage = downloadImage;