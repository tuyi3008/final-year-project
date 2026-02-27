// photos.js - My Photos Page Functionality

console.log('🚀 photos.js loaded');
console.log('🔑 Token exists:', !!localStorage.getItem('token'));
console.log('👤 Auth object exists:', !!window.auth);

let currentUser = null;
let albums = [];
let currentAlbum = null;
let selectedPhotos = [];
let isSelectionMode = false;

async function initializePage() {
    console.log('🔄 Initializing page...');
    const isAuthenticated = await checkAuth();
    if (isAuthenticated) {
        loadAlbums();
        setupEventListeners();
        checkForChallengeSelection();
    } else {
        clearUserData();
        showLoginPrompt();
    }
}

function clearUserData() {
    console.log('🧹 Clearing user data');
    currentUser = null;
    albums = [];
    currentAlbum = null;
    selectedPhotos = [];
    isSelectionMode = false;
}

function showLoginPrompt() {
    console.log('🔐 Showing login prompt');
    const container = document.getElementById('albums-container');
    if (container) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="bi bi-person-circle"></i>
                <h4>Please Log In</h4>
                <p class="text-muted">You need to be logged in to view your photos</p>
                <button class="btn btn-gradient" onclick="window.auth?.showLoginModal()">
                    <i class="bi bi-box-arrow-in-right me-2"></i>
                    Sign In
                </button>
            </div>
        `;
    }

    const albumsView = document.getElementById('albums-container')?.parentElement;
    if (albumsView) albumsView.style.display = 'block';
    
    const selectedView = document.getElementById('selected-album-view');
    if (selectedView) selectedView.style.display = 'none';
}

document.addEventListener('DOMContentLoaded', initializePage);

document.addEventListener('userLoggedIn', async () => {
    console.log('🔥 User logged in event received, reloading albums...');
    await initializePage();
});

document.addEventListener('userLoggedOut', () => {
    console.log('👋 User logged out event received');
    clearUserData();
    showLoginPrompt();
    updateUIForLoggedOutUser();
});

function updateUIForLoggedOutUser() {
    document.querySelector('.btn-login').style.display = 'inline-block';
    document.querySelector('.user-menu').style.display = 'none';
}

async function checkAuth() {
    console.log('🔍 checkAuth started');

    const token = localStorage.getItem('token');
    console.log('📦 Token from localStorage:', token ? `Found (${token.substring(0,15)}...)` : 'Not found');
    
    if (!token) {
        console.log('❌ No token');
        return false;
    }

    try {
        console.log('📤 Sending request to /profile with header:', 'Bearer ' + token.substring(0,15) + '...');
        
        const response = await fetch('http://localhost:8000/profile', {
            method: 'GET',
            headers: new Headers({
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
        });
        
        console.log('📥 Response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return false;
        }
        
        const data = await response.json();
        console.log('📦 Response data:', data);
        
        if (data.code === 200) {
            console.log('✅ Auth success');
            currentUser = data;
            updateUIForLoggedInUser();
            return true;
        } else {
            console.log('❌ Auth failed');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return false;
        }
    } catch (error) {
        console.error('❌ Fetch error:', error);
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return false;
    }
}

function updateUIForLoggedInUser() {
    document.querySelector('.btn-login').style.display = 'none';
    document.querySelector('.user-menu').style.display = 'flex';
}

function setupEventListeners() {
    const createAlbumForm = document.getElementById('create-album-form');
    if (createAlbumForm) createAlbumForm.addEventListener('submit', createAlbum);
    
    const uploadImagesForm = document.getElementById('upload-images-form');
    if (uploadImagesForm) uploadImagesForm.addEventListener('submit', uploadImages);
    
    const imageFiles = document.getElementById('image-files');
    if (imageFiles) imageFiles.addEventListener('change', previewImages);
    
    const backToAlbums = document.getElementById('back-to-albums');
    if (backToAlbums) backToAlbums.addEventListener('click', showAlbumsView);
    
    const deleteAlbumBtn = document.getElementById('delete-album-btn');
    if (deleteAlbumBtn) deleteAlbumBtn.addEventListener('click', deleteCurrentAlbum);
    
    const editAlbumForm = document.getElementById('edit-album-form');
    if (editAlbumForm) editAlbumForm.addEventListener('submit', updateAlbum);
    
    const downloadPhotoBtn = document.getElementById('download-photo-btn');
    if (downloadPhotoBtn) downloadPhotoBtn.addEventListener('click', downloadCurrentPhoto);
    
    const deletePhotoBtn = document.getElementById('delete-photo-btn');
    if (deletePhotoBtn) deletePhotoBtn.addEventListener('click', deleteCurrentPhoto);
}

async function loadAlbums() {
    console.log('📚 Loading albums from backend...');
    
    try {
        const token = localStorage.getItem('token');
        if (!token) {
            console.log('❌ No token for albums request');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }

        const response = await fetch('http://localhost:8000/api/albums', {
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            }
        });
        
        console.log('📥 Albums response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        console.log('📦 Albums data:', data);
        
        if (data.code === 200) {
            albums = data.albums;
            renderAlbums();
        } else {
            console.error('Failed to load albums:', data.error);
            showEmptyState();
        }
    } catch (error) {
        console.error('Error loading albums:', error);
        showEmptyState();
    }
}

function renderAlbums() {
    const container = document.getElementById('albums-container');
    
    if (!albums || albums.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="bi bi-journal-album"></i>
                <h4>No Albums Yet</h4>
                <p class="text-muted">Create your first album to start organizing your photos</p>
                <button class="btn btn-gradient" data-bs-toggle="modal" data-bs-target="#createAlbumModal">
                    <i class="bi bi-plus-circle me-2"></i>
                    Create Album
                </button>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    albums.forEach(album => {
        const albumCard = createAlbumCard(album);
        container.appendChild(albumCard);
    });
}

function createAlbumCard(album) {
    const card = document.createElement('div');
    card.className = 'album-card';
    card.dataset.albumId = album.id;
    
    const photoCount = album.photo_count || 0;
    const createdDate = new Date(album.created_at).toLocaleDateString();

    let coverHtml;
    if (album.cover_image) {
        const imagePath = album.cover_image.startsWith('/') ? album.cover_image : '/' + album.cover_image;
        coverHtml = `<img src="${imagePath}" alt="${album.name}" onerror="this.style.display='none';this.parentElement.innerHTML='<div class=\'album-cover-placeholder\'><i class=\'bi bi-images\'></i></div>'">`;
    } else {
        const gradient = getRandomAlbumCover();
        coverHtml = `
            <div class="album-cover-placeholder" style="background: ${gradient}">
                <i class="bi bi-images"></i>
            </div>
        `;
    }
    
    card.innerHTML = `
        <div class="album-cover">
            ${coverHtml}
        </div>
        <div class="album-info">
            <div class="album-header">
                <h3 class="album-title">${album.name}</h3>
                <div class="album-actions">
                    <button class="album-action-btn" onclick="editAlbum('${album.id}')" title="Edit">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="album-action-btn" onclick="deleteAlbum('${album.id}')" title="Delete">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
            ${album.description ? `<p class="album-description">${album.description}</p>` : ''}
            <div class="album-meta">
                <span><i class="bi bi-images"></i> ${photoCount} photos</span>
                <span><i class="bi bi-calendar3"></i> ${createdDate}</span>
            </div>
        </div>
    `;
    
    card.addEventListener('click', (e) => {
        if (!e.target.closest('.album-action-btn')) {
            openAlbum(album.id);
        }
    });
    
    return card;
}

function getRandomAlbumCover() {
    const gradients = [
        'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
    ];
    return gradients[Math.floor(Math.random() * gradients.length)];
}

async function openAlbum(albumId) {
    console.log('📂 Opening album:', albumId);
    
    try {
        const token = localStorage.getItem('token');
        if (!token) {
            console.log('❌ No token');
            return;
        }

        const response = await fetch(`http://localhost:8000/api/albums/${albumId}`, {
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            }
        });
        
        console.log('📥 Album response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            currentAlbum = data.album;
            renderAlbumView();
            checkForChallengeSelection();
        } else {
            console.error('Failed to load album:', data.error);
        }
    } catch (error) {
        console.error('Error opening album:', error);
    }
}

function renderAlbumView() {
    document.getElementById('albums-container').parentElement.style.display = 'none';
    document.getElementById('selected-album-view').style.display = 'block';
    
    document.getElementById('selected-album-title').textContent = currentAlbum.name;
    document.getElementById('album-photo-count').textContent = currentAlbum.photos?.length || 0;
    document.getElementById('album-created-date').textContent = new Date(currentAlbum.created_at).toLocaleDateString();
    document.getElementById('album-last-updated').textContent = new Date(currentAlbum.updated_at).toLocaleDateString();
    
    renderPhotos(currentAlbum.photos || []);
}

function renderPhotos(photos) {
    const container = document.getElementById('photos-container');
    
    if (!photos || photos.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="bi bi-images"></i>
                <h4>No Photos Yet</h4>
                <p class="text-muted">Upload your first image to this album</p>
                <button class="btn btn-gradient" data-bs-toggle="modal" data-bs-target="#uploadImageModal">
                    <i class="bi bi-upload me-2"></i>
                    Upload Images
                </button>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    photos.forEach(photo => {
        const photoCard = createPhotoCard(photo);
        container.appendChild(photoCard);
    });
}

function createPhotoCard(photo) {
    const card = document.createElement('div');
    card.className = 'photo-card';
    card.dataset.photoId = photo.id;
    
    const uploadDate = new Date(photo.uploaded_at).toLocaleDateString();
    
    card.innerHTML = `
        ${isSelectionMode ? 
            `<input type="checkbox" class="photo-checkbox" ${selectedPhotos.includes(photo.id) ? 'checked' : ''}>` : 
            ''
        }
        <img src="${photo.image_path}" alt="${photo.filename}">
        <div class="photo-overlay">
            <div class="photo-filename">${photo.filename}</div>
            <div class="photo-meta">
                <span><i class="bi bi-calendar3"></i> ${uploadDate}</span>
                <span class="photo-style">${photo.style || 'Original'}</span>
            </div>
        </div>
    `;
    
    card.addEventListener('click', (e) => {
        if (isSelectionMode && e.target.type === 'checkbox') {
            togglePhotoSelection(photo.id);
        } else if (!isSelectionMode) {
            openPhotoDetail(photo);
        }
    });
    
    if (isSelectionMode) {
        const checkbox = card.querySelector('.photo-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                togglePhotoSelection(photo.id, e.target.checked);
            });
        }
    }
    
    return card;
}

function openPhotoDetail(photo) {
    document.getElementById('detail-photo-image').src = photo.image_path;
    document.getElementById('detail-photo-filename').textContent = photo.filename;
    document.getElementById('detail-photo-date').textContent = new Date(photo.uploaded_at).toLocaleString();
    document.getElementById('detail-photo-size').textContent = formatFileSize(photo.file_size);
    document.getElementById('detail-photo-style').textContent = photo.style || 'Original';
    
    document.getElementById('detail-photo-image').dataset.photoId = photo.id;
    document.getElementById('detail-photo-image').dataset.photoPath = photo.image_path;
    
    const modal = new bootstrap.Modal(document.getElementById('photoDetailModal'));
    modal.show();
}

function togglePhotoSelection(photoId, selected) {
    if (selected === undefined) {
        const index = selectedPhotos.indexOf(photoId);
        if (index === -1) {
            selectedPhotos.push(photoId);
        } else {
            selectedPhotos.splice(index, 1);
        }
    } else {
        if (selected) {
            if (!selectedPhotos.includes(photoId)) {
                selectedPhotos.push(photoId);
            }
        } else {
            selectedPhotos = selectedPhotos.filter(id => id !== photoId);
        }
    }
    
    updateSelectionBar();
}

function updateSelectionBar() {
    let bar = document.querySelector('.selection-bar');
    
    if (selectedPhotos.length > 0) {
        if (!bar) {
            bar = document.createElement('div');
            bar.className = 'selection-bar';
            document.getElementById('selected-album-view').insertBefore(bar, document.getElementById('photos-container'));
        }
        
        const isChallengeMode = window.location.search.includes('select=challenge');
        
        if (isChallengeMode) {
            bar.innerHTML = `
                <span><strong>${selectedPhotos.length}</strong> photos selected for challenge</span>
                <div class="d-flex gap-2">
                    <button class="btn btn-warning btn-sm" onclick="submitSelectedToChallenge()">
                        <i class="bi bi-trophy me-2"></i>Submit to Challenge
                    </button>
                    <button class="btn btn-outline-light btn-sm" onclick="exitSelectionMode()">
                        <i class="bi bi-x-lg"></i>Cancel
                    </button>
                </div>
            `;
        } else {
            bar.innerHTML = `
                <span><strong>${selectedPhotos.length}</strong> photos selected</span>
                <div class="d-flex gap-2">
                    <button class="btn btn-outline-light btn-sm" onclick="downloadSelected()">
                        <i class="bi bi-download"></i> Download
                    </button>
                    <button class="btn btn-outline-danger btn-sm" onclick="deleteSelected()">
                        <i class="bi bi-trash"></i> Delete
                    </button>
                    <button class="btn btn-outline-light btn-sm" onclick="exitSelectionMode()">
                        <i class="bi bi-x-lg"></i> Cancel
                    </button>
                </div>
            `;
        }
    } else if (bar) {
        bar.remove();
        exitSelectionMode();
    }
}

function enterSelectionMode() {
    isSelectionMode = true;
    if (currentAlbum && currentAlbum.photos) {
        renderPhotos(currentAlbum.photos);
    }
}

function exitSelectionMode() {
    isSelectionMode = false;
    selectedPhotos = [];
    if (currentAlbum && currentAlbum.photos) {
        renderPhotos(currentAlbum.photos);
    }
    
    const bar = document.querySelector('.selection-bar');
    if (bar) bar.remove();
}

function showAlbumsView() {
    document.getElementById('albums-container').parentElement.style.display = 'block';
    document.getElementById('selected-album-view').style.display = 'none';
    currentAlbum = null;
    exitSelectionMode();
}

function checkForChallengeSelection() {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('select') === 'challenge') {
        setTimeout(() => {
            enterChallengeSelectionMode();
        }, 500);
    }
}

function enterChallengeSelectionMode() {
    if (!currentAlbum) {
        alert('Please open an album first to select photos for challenge');
        return;
    }
    
    if (!isSelectionMode) {
        enterSelectionMode();
    }
}

async function submitSelectedToChallenge() {
    if (selectedPhotos.length === 0) {
        alert('Please select at least one photo');
        return;
    }
    
    const pendingChallenge = sessionStorage.getItem('pendingChallenge');
    if (!pendingChallenge) {
        alert('No active challenge found');
        return;
    }
    
    const challenge = JSON.parse(pendingChallenge);
    const token = localStorage.getItem('token');
    
    const selectedItems = currentAlbum.photos.filter(p => selectedPhotos.includes(p.id));
    
    if (selectedItems.length === 0) return;
    
    if (selectedItems.length === 1) {
        const photo = selectedItems[0];
        showChallengeSubmitModal(photo, challenge);
    } else {
        showPhotoSelectionModal(selectedItems, challenge);
    }
}

function showChallengeSubmitModal(photo, challenge) {
    const modalHtml = `
        <div class="modal fade" id="challengeSubmitModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content glass-card">
                    <div class="modal-header border-0">
                        <h5 class="modal-title">
                            <i class="bi bi-trophy-fill text-warning me-2"></i>
                            Submit to Challenge: ${challenge.theme}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="text-center mb-4">
                            <img src="${photo.image_path}" class="img-fluid rounded" style="max-height: 200px;">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Challenge Theme</label>
                            <input type="text" class="form-control" value="${challenge.theme}" readonly>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Style</label>
                            <input type="text" class="form-control" value="${photo.style || 'Original'}" readonly>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Description</label>
                            <textarea class="form-control" id="challenge-description" rows="3" placeholder="Tell us about your creation..."></textarea>
                        </div>
                        
                        <div class="d-grid gap-2">
                            <button class="btn btn-gradient" onclick="confirmChallengeSubmit('${photo.id}', '${photo.image_path}', '${photo.style || 'Original'}')">
                                <i class="bi bi-cloud-upload me-2"></i>
                                Submit to Challenge
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('challengeSubmitModal'));
    modal.show();
    
    document.getElementById('challengeSubmitModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

function showPhotoSelectionModal(photos, challenge) {
    let photosHtml = '';
    photos.forEach(photo => {
        photosHtml += `
            <div class="col-6 mb-2">
                <div class="card bg-dark border-primary photo-select-card" onclick="selectPhotoForChallenge('${photo.id}', '${photo.image_path}', '${photo.style || 'Original'}')">
                    <img src="${photo.image_path}" class="card-img-top" style="aspect-ratio: 1; object-fit: cover;">
                    <div class="card-body p-2 text-center">
                        <small>${photo.style || 'Original'}</small>
                    </div>
                </div>
            </div>
        `;
    });
    
    const modalHtml = `
        <div class="modal fade" id="photoSelectModal" tabindex="-1">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content glass-card">
                    <div class="modal-header border-0">
                        <h5 class="modal-title">
                            <i class="bi bi-trophy-fill text-warning me-2"></i>
                            Select Photo for Challenge
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p class="text-muted mb-3">Choose which photo to submit to "${challenge.theme}"</p>
                        <div class="row g-2">
                            ${photosHtml}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('photoSelectModal'));
    modal.show();
    
    document.getElementById('photoSelectModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

function selectPhotoForChallenge(photoId, imagePath, style) {
    const modal = bootstrap.Modal.getInstance(document.getElementById('photoSelectModal'));
    modal.hide();
    
    const pendingChallenge = sessionStorage.getItem('pendingChallenge');
    if (pendingChallenge) {
        const challenge = JSON.parse(pendingChallenge);
        showChallengeSubmitModal({ id: photoId, image_path: imagePath, style: style }, challenge);
    }
}

async function confirmChallengeSubmit(photoId, imagePath, style) {
    const description = document.getElementById('challenge-description').value;
    const token = localStorage.getItem('token');
    const pendingChallenge = sessionStorage.getItem('pendingChallenge');
    
    if (!description.trim()) {
        alert('Please add a description');
        return;
    }
    
    if (!pendingChallenge) {
        alert('No active challenge found');
        return;
    }
    
    const challenge = JSON.parse(pendingChallenge);
    
    const submitBtn = document.querySelector('#challengeSubmitModal .btn-gradient');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Submitting...';
    
    try {
        const response = await fetch('/api/challenge/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token
            },
            body: JSON.stringify({
                image_path: imagePath,
                style: style,
                description: description,
                challenge_id: challenge.id
            })
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            alert('Successfully submitted to challenge!');
            
            const modal = bootstrap.Modal.getInstance(document.getElementById('challengeSubmitModal'));
            modal.hide();
            
            sessionStorage.removeItem('pendingChallenge');
            
            exitSelectionMode();
            
            const urlParams = new URLSearchParams(window.location.search);
            urlParams.delete('select');
            const newUrl = window.location.pathname;
            window.history.replaceState({}, '', newUrl);
        } else {
            alert(data.error || 'Failed to submit');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to submit. Please try again.');
    }
}

async function createAlbum(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('token');
    console.log('📦 Create album token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for create album');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    const formData = new FormData();
    formData.append('name', document.getElementById('album-name').value);
    formData.append('description', document.getElementById('album-description').value);
    
    const coverFile = document.getElementById('album-cover').files[0];
    if (coverFile) {
        formData.append('cover', coverFile);
    }
    
    const submitBtn = document.getElementById('create-album-submit');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Creating...';
    
    try {
        const response = await fetch('http://localhost:8000/api/albums', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token
            },
            body: formData
        });
        
        console.log('📥 Create album response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            const modal = bootstrap.Modal.getInstance(document.getElementById('createAlbumModal'));
            if (modal) modal.hide();
            
            loadAlbums();
            document.getElementById('create-album-form').reset();
            showSuccessMessage('Album created successfully!');
        } else {
            alert(data.error || 'Failed to create album');
        }
    } catch (error) {
        console.error('Error creating album:', error);
        alert('Failed to create album');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="bi bi-check-circle me-2"></i>Create Album';
    }
}

async function uploadImages(e) {
    e.preventDefault();
    
    const token = localStorage.getItem('token');
    console.log('📤 Upload images token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for upload');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    const files = document.getElementById('image-files').files;
    if (files.length === 0) return;
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }
    formData.append('album_id', currentAlbum.id);
    
    const submitBtn = document.getElementById('upload-images-submit');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
    
    try {
        const response = await fetch('http://localhost:8000/api/albums/upload', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token
            },
            body: formData
        });
        
        console.log('📥 Upload response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            bootstrap.Modal.getInstance(document.getElementById('uploadImageModal')).hide();
            openAlbum(currentAlbum.id);
            document.getElementById('upload-images-form').reset();
            document.getElementById('image-preview-container').innerHTML = '';
            showSuccessMessage('Images uploaded successfully!');
        } else {
            alert(data.error || 'Failed to upload images');
        }
    } catch (error) {
        console.error('Error uploading images:', error);
        alert('Failed to upload images');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="bi bi-cloud-upload me-2"></i>Upload to Album';
    }
}

function previewImages() {
    const files = document.getElementById('image-files').files;
    const container = document.getElementById('image-preview-container');
    
    container.innerHTML = '';
    
    for (let i = 0; i < Math.min(files.length, 8); i++) {
        const file = files[i];
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const preview = document.createElement('div');
            preview.className = 'preview-item';
            preview.innerHTML = `
                <img src="${e.target.result}" alt="Preview">
                <button class="remove-preview" onclick="this.parentElement.remove()">
                    <i class="bi bi-x"></i>
                </button>
            `;
            container.appendChild(preview);
        };
        
        reader.readAsDataURL(file);
    }
    
    if (files.length > 8) {
        const more = document.createElement('div');
        more.className = 'preview-item d-flex align-items-center justify-content-center';
        more.style.background = 'rgba(255,255,255,0.1)';
        more.innerHTML = `+${files.length - 8} more`;
        container.appendChild(more);
    }
}

async function editAlbum(albumId) {
    const album = albums.find(a => a.id === albumId);
    if (!album) return;
    
    document.getElementById('edit-album-id').value = album.id;
    document.getElementById('edit-album-name').value = album.name;
    document.getElementById('edit-album-description').value = album.description || '';
    
    const modal = new bootstrap.Modal(document.getElementById('editAlbumModal'));
    modal.show();
}

async function updateAlbum(e) {
    e.preventDefault();

    const token = localStorage.getItem('token');
    console.log('✏️ Update album token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for update');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    const albumId = document.getElementById('edit-album-id').value;
    
    try {
        const response = await fetch(`http://localhost:8000/api/albums/${albumId}`, {
            method: 'PUT',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: document.getElementById('edit-album-name').value,
                description: document.getElementById('edit-album-description').value
            })
        });
        
        console.log('📥 Update album response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            const modal = bootstrap.Modal.getInstance(document.getElementById('editAlbumModal'));
            if (modal) modal.hide();

            loadAlbums();
            
            if (currentAlbum && currentAlbum.id === albumId) {
                openAlbum(albumId);
            }

            showSuccessMessage('Album updated successfully!');
        } else {
            alert(data.error || 'Failed to update album');
        }
    } catch (error) {
        console.error('Error updating album:', error);
        alert('Failed to update album');
    }
}

async function deleteAlbum(albumId) {
    const token = localStorage.getItem('token');
    console.log('🗑️ Delete album token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for delete');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    if (!confirm('Are you sure you want to delete this album? All photos will be deleted.')) {
        return;
    }
    
    try {
        const response = await fetch(`http://localhost:8000/api/albums/${albumId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': 'Bearer ' + token
            }
        });
        
        console.log('📥 Delete album response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            if (currentAlbum && currentAlbum.id === albumId) {
                showAlbumsView();
            }

            loadAlbums();
            showSuccessMessage('Album deleted successfully!');
        } else {
            alert(data.error || 'Failed to delete album');
        }
    } catch (error) {
        console.error('Error deleting album:', error);
        alert('Failed to delete album');
    }
}

function deleteCurrentAlbum() {
    if (currentAlbum) {
        deleteAlbum(currentAlbum.id);
    }
}

async function deletePhoto(photoId) {
    const token = localStorage.getItem('token');
    console.log('🗑️ Delete photo token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for delete photo');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    if (!confirm('Are you sure you want to delete this photo?')) {
        return;
    }
    
    try {
        const response = await fetch(`http://localhost:8000/api/photos/${photoId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': 'Bearer ' + token
            }
        });
        
        console.log('📥 Delete photo response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            bootstrap.Modal.getInstance(document.getElementById('photoDetailModal')).hide();
            openAlbum(currentAlbum.id);
            showSuccessMessage('Photo deleted successfully!');
        } else {
            alert(data.error || 'Failed to delete photo');
        }
    } catch (error) {
        console.error('Error deleting photo:', error);
        alert('Failed to delete photo');
    }
}

function deleteCurrentPhoto() {
    const photoId = document.getElementById('detail-photo-image').dataset.photoId;
    if (photoId) {
        deletePhoto(photoId);
    }
}

async function deleteSelected() {
    if (selectedPhotos.length === 0) return;
    
    const token = localStorage.getItem('token');
    console.log('🗑️ Delete selected photos token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for delete selected');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    if (!confirm(`Delete ${selectedPhotos.length} selected photos?`)) {
        return;
    }
    
    try {
        const response = await fetch('http://localhost:8000/api/photos/batch-delete', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ photo_ids: selectedPhotos })
        });
        
        console.log('📥 Delete selected response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const data = await response.json();
        
        if (data.code === 200) {
            exitSelectionMode();
            openAlbum(currentAlbum.id);
            showSuccessMessage(`${selectedPhotos.length} photos deleted successfully!`);
        } else {
            alert(data.error || 'Failed to delete photos');
        }
    } catch (error) {
        console.error('Error deleting photos:', error);
        alert('Failed to delete photos');
    }
}

function downloadPhoto(photoPath, filename) {
    const link = document.createElement('a');
    link.href = photoPath;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadCurrentPhoto() {
    const photoPath = document.getElementById('detail-photo-image').dataset.photoPath;
    const filename = document.getElementById('detail-photo-filename').textContent;
    if (photoPath && filename) {
        downloadPhoto(photoPath, filename);
    }
}

async function downloadSelected() {
    if (selectedPhotos.length === 0) return;
    
    const token = localStorage.getItem('token');
    console.log('📥 Download selected token:', token ? 'Present' : 'Missing');
    
    if (!token) {
        console.log('❌ No token for download');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    try {
        const response = await fetch('http://localhost:8000/api/photos/batch-download', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ photo_ids: selectedPhotos })
        });
        
        console.log('📥 Download selected response status:', response.status);
        
        if (response.status === 401) {
            console.log('⚠️ Token invalid');
            localStorage.removeItem('token');
            if (window.auth?.showLoginModal) {
                window.auth.showLoginModal();
            }
            return;
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `album_${currentAlbum.id}_photos.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error downloading photos:', error);
        alert('Failed to download photos');
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

function showEmptyState() {
    const container = document.getElementById('albums-container');
    container.innerHTML = `
        <div class="empty-state">
            <i class="bi bi-journal-album"></i>
            <h4>No Albums Yet</h4>
            <p class="text-muted">Create your first album to start organizing your photos</p>
            <button class="btn btn-gradient" data-bs-toggle="modal" data-bs-target="#createAlbumModal">
                <i class="bi bi-plus-circle me-2"></i>
                Create Album
            </button>
        </div>
    `;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

window.editAlbum = editAlbum;
window.deleteAlbum = deleteAlbum;
window.downloadSelected = downloadSelected;
window.deleteSelected = deleteSelected;
window.exitSelectionMode = exitSelectionMode;
window.enterSelectionMode = enterSelectionMode;
window.submitSelectedToChallenge = submitSelectedToChallenge;
window.selectPhotoForChallenge = selectPhotoForChallenge;
window.confirmChallengeSubmit = confirmChallengeSubmit;