// profile.js

// ====================== User Profile Functions ======================

function loadUserProfile() {
    const userEmail = localStorage.getItem('userEmail');
    document.getElementById('profileName').textContent = userEmail.split('@')[0] || 'User';
    document.getElementById('profileEmail').textContent = userEmail;
    
    // Load saved bio if exists
    const savedBio = localStorage.getItem('userBio');
    if (savedBio) {
        document.getElementById('settingsBio').value = savedBio;
    }
    
    // Set join date (you can get this from backend later)
    const joinDate = localStorage.getItem('joinDate') || 'Jan 2026';
    document.getElementById('joinDate').textContent = joinDate;
    
    // Set last active
    document.getElementById('lastActive').textContent = 'Today';
    
    // Load level and XP data
    loadLevelData();
}

// ====================== Level System Functions ======================

function loadLevelData() {
    // Get XP from localStorage or default to 0
    let totalXP = parseInt(localStorage.getItem('userXP') || '0');
    
    // Calculate level based on XP (simple formula: level = floor(xp/100) + 1)
    const level = Math.floor(totalXP / 100) + 1;
    const currentLevelXP = (level - 1) * 100;
    const nextLevelXP = level * 100;
    const progress = ((totalXP - currentLevelXP) / 100) * 100;

    const userLevelEl = document.getElementById('userLevel');
    if (userLevelEl) userLevelEl.textContent = level;
    
    const levelDisplayEl = document.getElementById('levelDisplay');
    if (levelDisplayEl) levelDisplayEl.textContent = level;
    
    const levelValueEl = document.getElementById('levelValue');
    if (levelValueEl) levelValueEl.textContent = level;
    
    const totalPointsEl = document.getElementById('totalPoints');
    if (totalPointsEl) totalPointsEl.textContent = totalXP;
    
    const currentExpEl = document.getElementById('currentExp');
    if (currentExpEl) currentExpEl.textContent = totalXP - currentLevelXP;
    
    const nextLevelExpEl = document.getElementById('nextLevelExp');
    if (nextLevelExpEl) nextLevelExpEl.textContent = '100';
    
    const nextLevelPointsEl = document.getElementById('nextLevelPoints');
    if (nextLevelPointsEl) nextLevelPointsEl.textContent = nextLevelXP;
    
    const expProgressEl = document.getElementById('expProgress');
    if (expProgressEl) expProgressEl.style.width = progress + '%';
}

// ====================== Stats Functions ======================

async function loadUserStats() {
    console.log('📊 Loading user stats from backend...');
    
    const token = localStorage.getItem('token');
    if (!token) {
        console.log('❌ No token for stats');
        return;
    }
    
    try {
        const response = await fetch('http://localhost:8000/api/user/stats', {
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            }
        });
        
        console.log('📥 Stats response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        console.log('📦 Stats data:', data);
        
        if (data.code === 200) {
            document.getElementById('transformCount').textContent = data.transformCount || '0';
            document.getElementById('favoriteCount').textContent = data.favoriteCount || '0';
            document.getElementById('shareCount').textContent = data.shareCount || '0';
            
            // Update progress bars based on stats
            updateProgressBars(data);
            
            // Update XP based on stats (example calculation)
            const xpFromStats = (data.transformCount * 5) + (data.favoriteCount * 2) + (data.shareCount * 10);
            localStorage.setItem('userXP', xpFromStats.toString());
            loadLevelData();
            
        } else {
            console.error('Failed to load stats:', data.error);
            setDefaultStats();
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        setDefaultStats();
    }
}

function setDefaultStats() {
    document.getElementById('transformCount').textContent = '0';
    document.getElementById('favoriteCount').textContent = '0';
    document.getElementById('shareCount').textContent = '0';
}

function updateProgressBars(data) {
    // Example: set goals (you can adjust these numbers)
    const transformGoal = 50;
    const favoriteGoal = 20;
    const shareGoal = 10;
    
    document.getElementById('transformProgress').style.width = 
        Math.min((data.transformCount / transformGoal) * 100, 100) + '%';
    document.getElementById('favoriteProgress').style.width = 
        Math.min((data.favoriteCount / favoriteGoal) * 100, 100) + '%';
    document.getElementById('shareProgress').style.width = 
        Math.min((data.shareCount / shareGoal) * 100, 100) + '%';
}

// ====================== History Functions ======================

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
        
        const response = await fetch('http://localhost:8000/history', {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        
        const data = await response.json();
        
        if (data.code === 200 && data.history && data.history.length > 0) {
            const historyWithFav = await Promise.all(
                data.history.map(async (item) => {
                    const isFav = await checkFavoriteStatus(item.result_image_path);
                    return { ...item, isFavorite: isFav };
                })
            );
            displayHistory(historyWithFav);
        } else {
            showEmptyHistory();
        }
    } catch (error) {
        console.error('Error loading history:', error);
        showEmptyHistory();
    }
}

function showEmptyHistory() {
    const historyContainer = document.getElementById('historyContainer');
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
                            <button class="btn-favorite ${isFav ? 'active' : ''}" 
                                    onclick="toggleFavorite('${item.result_image_path}', '${item.style}', this)">
                                <i class="bi ${isFav ? 'bi-heart-fill' : 'bi-heart'}"></i>
                            </button>
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

// ====================== Favorites Functions ======================

async function loadFavorites() {
    console.log('Loading favorites...');
    
    const favoritesContainer = document.getElementById('favoritesContainer');
    const token = localStorage.getItem('token');
    
    if (!token) return;
    
    try {
        const response = await fetch('http://localhost:8000/favorites', {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        
        const data = await response.json();
        
        if (data.code === 200 && data.favorites && data.favorites.length > 0) {
            displayFavorites(data.favorites);
        } else {
            showEmptyFavorites();
        }
    } catch (error) {
        console.error('Error loading favorites:', error);
        showEmptyFavorites();
    }
}

function showEmptyFavorites() {
    const favoritesContainer = document.getElementById('favoritesContainer');
    favoritesContainer.innerHTML = `
        <div class="col-12 text-center py-5 text-muted">
            <i class="bi bi-heart display-4 mb-3"></i>
            <p>No favorites yet</p>
        </div>
    `;
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

// ====================== Favorite Toggle Function ======================

async function checkFavoriteStatus(imagePath) {
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`http://localhost:8000/favorites/check?image_path=${encodeURIComponent(imagePath)}`, {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        const data = await response.json();
        return data.is_favorite || false;
    } catch (error) {
        return false;
    }
}

async function toggleFavorite(imagePath, style, button) {
    const token = localStorage.getItem('token');
    const isFavorite = button.classList.contains('active');
    const icon = button.querySelector('i');
    
    try {
        const response = await fetch(isFavorite ? 'http://localhost:8000/favorites/remove' : 'http://localhost:8000/favorites/add', {
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
            
            // Update favorite count
            const favCount = document.getElementById('favoriteCount');
            if (favCount) {
                favCount.textContent = parseInt(favCount.textContent) + (isFavorite ? -1 : 1);
            }
            
            // Update XP for getting favorites (if this was a like received)
            if (!isFavorite) {
                addXP(2); // +2 XP for receiving a like
            }
            
            // Refresh favorites tab if active
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

// ====================== XP System Functions ======================

function addXP(amount) {
    let currentXP = parseInt(localStorage.getItem('userXP') || '0');
    currentXP += amount;
    localStorage.setItem('userXP', currentXP.toString());
    loadLevelData();
    
    // Show XP gained notification
    showXPMessage(`+${amount} XP`);
}

function showXPMessage(message) {
    const xpDiv = document.createElement('div');
    xpDiv.className = 'alert alert-info alert-dismissible fade show position-fixed top-0 end-0 m-3';
    xpDiv.style.zIndex = '9999';
    xpDiv.innerHTML = `
        <i class="bi bi-star-fill text-warning me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(xpDiv);
    
    setTimeout(() => {
        xpDiv.remove();
    }, 2000);
}

// ====================== Download Function ======================

function downloadImage(imagePath) {
    const link = document.createElement('a');
    link.href = `/${imagePath}`;
    link.download = imagePath.split('/').pop();
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Show download message
    showDownloadMessage();
}

function showDownloadMessage() {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        <i class="bi bi-check-circle-fill me-2"></i>
        Image downloaded successfully!
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 3000);
}

// ====================== Edit Profile Functions ======================

function loadUserDataForEdit() {
    const username = document.getElementById('profileName').textContent;
    const bio = localStorage.getItem('userBio') || '';
    
    document.getElementById('editUsername').value = username;
    document.getElementById('editBio').value = bio;
}

async function handleEditProfile(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('token');
    if (!token) {
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal('Please login to edit profile');
        }
        return;
    }
    
    const username = document.getElementById('editUsername').value;
    const bio = document.getElementById('editBio').value;
    const avatarFile = document.getElementById('editAvatar')?.files[0];
    
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Saving...';
    
    try {

        const formData = new FormData();
        formData.append('username', username);
        formData.append('bio', bio);
        if (avatarFile) {
            formData.append('avatar', avatarFile);
        }

        const response = await fetch('http://localhost:8000/api/user/profile', {
            method: 'PUT',
            headers: {
                'Authorization': 'Bearer ' + token
            },
            body: formData
        });
        
        if (response.status === 401) {
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal('Session expired. Please login again.');
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {

            document.getElementById('profileName').textContent = username;
            localStorage.setItem('userBio', bio);

            if (data.avatar_path) {
                updateAvatarDisplay(data.avatar_path);
            }

            const modal = bootstrap.Modal.getInstance(document.getElementById('editProfileModal'));
            modal.hide();
            
            showSuccessMessage('Profile updated successfully!');
            addXP(10);
        } else {
            alert(data.error || 'Failed to update profile');
        }
    } catch (error) {
        console.error('Error updating profile:', error);
        alert('Failed to update profile. Please try again.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

function previewAvatar(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {

            let preview = document.getElementById('avatarPreview');
            if (!preview) {

                const avatarContainer = document.querySelector('.profile-avatar');
                if (avatarContainer) {

                    const originalIcon = avatarContainer.innerHTML;

                    preview = document.createElement('img');
                    preview.id = 'avatarPreview';
                    preview.className = 'avatar-preview';
                    preview.style.width = '100%';
                    preview.style.height = '100%';
                    preview.style.borderRadius = '50%';
                    preview.style.objectFit = 'cover';

                    avatarContainer.innerHTML = '';
                    avatarContainer.appendChild(preview);

                    avatarContainer.dataset.originalIcon = originalIcon;
                }
            }
            
            if (preview) {
                preview.src = e.target.result;
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function updateAvatarDisplay(avatarPath) {
    const avatarContainer = document.querySelector('.profile-avatar');
    if (avatarContainer) {

        let avatarImg = avatarContainer.querySelector('img');
        if (!avatarImg) {

            avatarImg = document.createElement('img');
            avatarImg.className = 'avatar-preview';
            avatarImg.style.width = '100%';
            avatarImg.style.height = '100%';
            avatarImg.style.borderRadius = '50%';
            avatarImg.style.objectFit = 'cover';
            avatarContainer.innerHTML = '';
            avatarContainer.appendChild(avatarImg);
        }
        avatarImg.src = `/${avatarPath}`;
    }
}

async function loadUserAvatar() {
    const token = localStorage.getItem('token');
    if (!token) return;
    
    try {
        const response = await fetch('http://localhost:8000/api/user/profile', {
            headers: {
                'Authorization': 'Bearer ' + token
            }
        });
        
        const data = await response.json();
        if (data.code === 200 && data.avatar_path) {
            updateAvatarDisplay(data.avatar_path);
        }
    } catch (error) {
        console.error('Error loading avatar:', error);
    }
}

function showSuccessMessage(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        <i class="bi bi-check-circle-fill me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 3000);
}

// ====================== Settings Functions ======================

function setupSettingsTab() {
    // Load saved preferences
    const emailNotifications = localStorage.getItem('emailNotifications') !== 'false';
    const pushNotifications = localStorage.getItem('pushNotifications') !== 'false';
    const newsletter = localStorage.getItem('newsletter') !== 'false';
    
    document.getElementById('emailNotifications').checked = emailNotifications;
    document.getElementById('pushNotifications').checked = pushNotifications;
    document.getElementById('newsletter').checked = newsletter;
    
    // Add event listeners
    document.getElementById('emailNotifications').addEventListener('change', function(e) {
        localStorage.setItem('emailNotifications', e.target.checked);
    });
    
    document.getElementById('pushNotifications').addEventListener('change', function(e) {
        localStorage.setItem('pushNotifications', e.target.checked);
    });
    
    document.getElementById('newsletter').addEventListener('change', function(e) {
        localStorage.setItem('newsletter', e.target.checked);
    });
    
    // Settings form submit
    const settingsForm = document.getElementById('settingsForm');
    if (settingsForm) {
        settingsForm.addEventListener('submit', handleSettingsSubmit);
    }
    
    // Logout button
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
}

async function handleSettingsSubmit(e) {
    e.preventDefault();
    
    const username = document.getElementById('settingsUsername').value;
    const email = document.getElementById('settingsEmail').value;
    const bio = document.getElementById('settingsBio').value;
    
    if (username) {
        document.getElementById('profileName').textContent = username;
    }
    
    if (bio) {
        localStorage.setItem('userBio', bio);
    }
    
    showSuccessMessage('Settings saved successfully!');
    addXP(5); // +5 XP for updating settings
}

function handleLogout() {
    if (window.auth) {
        window.auth.logout();
    } else {
        localStorage.removeItem('token');
        localStorage.removeItem('userEmail');
        localStorage.removeItem('userXP');
        window.location.href = '/';
    }
}

// ====================== Initialization ======================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Profile page loaded');
    
    // Check if user is logged in
    if (!window.auth?.isLoggedIn()) {
        window.location.href = '/';
        return;
    }
    
    // Load all user data
    loadUserProfile();
    loadUserStats();
    loadHistory();
    loadFavorites();
    loadUserAvatar();

    const avatarInput = document.getElementById('editAvatar');
    if (avatarInput) {
        avatarInput.addEventListener('change', function() {
            previewAvatar(this);
        });
    }
    
    // Setup tab event listeners
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
    
    // Setup edit profile modal
    const editProfileForm = document.getElementById('editProfileForm');
    if (editProfileForm) {
        editProfileForm.addEventListener('submit', handleEditProfile);
    }
    
    const editProfileModal = document.getElementById('editProfileModal');
    if (editProfileModal) {
        editProfileModal.addEventListener('show.bs.modal', loadUserDataForEdit);
    }
    
    // Setup settings tab
    setupSettingsTab();
});

// Make functions global for onclick handlers
window.toggleFavorite = toggleFavorite;
window.downloadImage = downloadImage;