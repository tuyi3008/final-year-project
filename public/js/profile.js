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

function loadHistory() {
    // TODO: Fetch from backend
    const historyContainer = document.getElementById('historyContainer');
    
    // Sample history data
    const sampleHistory = [
        { id: 1, style: 'sketch', date: '2024-01-15', image: 'https://via.placeholder.com/80' },
        { id: 2, style: 'anime', date: '2024-01-14', image: 'https://via.placeholder.com/80' },
        { id: 3, style: 'ink', date: '2024-01-13', image: 'https://via.placeholder.com/80' }
    ];
    
    if (sampleHistory.length > 0) {
        historyContainer.innerHTML = '';
        sampleHistory.forEach(item => {
            historyContainer.innerHTML += `
                <div class="col-md-4">
                    <div class="history-item d-flex align-items-center gap-3">
                        <img src="${item.image}" alt="${item.style}" class="history-image">
                        <div>
                            <h6 class="mb-1">${item.style.charAt(0).toUpperCase() + item.style.slice(1)} Style</h6>
                            <p class="text-muted small mb-0">${item.date}</p>
                        </div>
                    </div>
                </div>
            `;
        });
    }
}