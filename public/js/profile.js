// profile.js

document.addEventListener('DOMContentLoaded', function() {
    console.log('Profile page loaded');
    
    // Check if user is logged in
    if (!auth.isLoggedIn()) {
        // Redirect to home if not logged in
        window.location.href = '/';
        return;
    }
    
    // Load user profile data
    loadUserProfile();
    
    // Load user stats
    loadUserStats();
    
    // Load history
    loadHistory();
});

function loadUserProfile() {
    // Get user email from localStorage
    const userEmail = localStorage.getItem('userEmail');
    
    // Update profile page
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
    
    if (!token) {
        console.log('No token found');
        return;
    }
    
    try {
        // Show loading state
        historyContainer.innerHTML = `
            <div class="col-12 text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3 text-muted">Loading your history...</p>
            </div>
        `;
        
        const response = await fetch('/history', {
            headers: {
                'Authorization': 'Bearer ' + token
            }
        });
        
        const data = await response.json();
        console.log('History data:', data);
        
        if (data.code === 200 && data.history && data.history.length > 0) {
            displayHistory(data.history);
        } else {
            // No history found
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
        historyContainer.innerHTML = `
            <div class="col-12 text-center py-5 text-muted">
                <i class="bi bi-exclamation-triangle display-4 mb-3"></i>
                <p>Failed to load history</p>
                <button onclick="loadHistory()" class="btn btn-outline-light mt-3">
                    <i class="bi bi-arrow-clockwise me-2"></i>Try Again
                </button>
            </div>
        `;
    }
}

function displayHistory(history) {
    const historyContainer = document.getElementById('historyContainer');
    historyContainer.innerHTML = '';
    
    history.forEach(item => {
        // Format date
        const date = new Date(item.created_at);
        const formattedDate = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        // Get image URL (you need to serve uploads folder statically)
        const imageUrl = item.result_image_path ? `/${item.result_image_path}` : 'https://via.placeholder.com/80';
        
        historyContainer.innerHTML += `
            <div class="col-md-4">
                <div class="history-item d-flex align-items-center gap-3">
                    <img src="${imageUrl}" alt="${item.style}" class="history-image">
                    <div>
                        <h6 class="mb-1">${item.style.charAt(0).toUpperCase() + item.style.slice(1)} Style</h6>
                        <p class="text-muted small mb-0">${formattedDate}</p>
                        <button class="btn btn-sm btn-outline-light mt-2" onclick="downloadImage('${item.result_image_path}')">
                            <i class="bi bi-download"></i> Download
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
}

function downloadImage(imagePath) {
    // Create a temporary link and click it to download
    const link = document.createElement('a');
    link.href = `/${imagePath}`;
    link.download = imagePath.split('/').pop();
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}