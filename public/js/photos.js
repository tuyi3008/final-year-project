// photos.js - My Photos Page Functionality

console.log('🚀 photos.js loaded');
console.log('🔑 Token exists:', !!localStorage.getItem('token'));
console.log('👤 Auth object exists:', !!window.auth);

let currentUser = null;
let albums = [];
let currentAlbum = null;
let selectedPhotos = [];
let isSelectionMode = false;

// Initialize page
document.addEventListener('DOMContentLoaded', async () => {
    const isAuthenticated = await checkAuth();
    if (isAuthenticated) {
        loadAlbums();
        setupEventListeners();
    }
});

console.log('🔥 photos.js loaded');
console.log('Token exists:', !!localStorage.getItem('token'));

async function checkAuth() {
    console.log('🔍 checkAuth started');

    const token = localStorage.getItem('token');
    console.log('📦 Token from localStorage:', token ? `Found (${token.substring(0,15)}...)` : 'Not found');
    
    if (!token) {
        console.log('❌ No token');
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal();
        }
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

        console.log('Request headers sent:', {
            'Authorization': 'Bearer ' + token.substring(0,10) + '...',
            'Content-Type': 'application/json'
        });
        
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

// Update UI for logged in user
function updateUIForLoggedInUser() {
    document.querySelector('.btn-login').style.display = 'none';
    document.querySelector('.user-menu').style.display = 'flex';
}

// Setup event listeners
function setupEventListeners() {
    // Create album form
    document.getElementById('create-album-form').addEventListener('submit', createAlbum);
    
    // Upload images form
    document.getElementById('upload-images-form').addEventListener('submit', uploadImages);
    
    // Image files preview
    document.getElementById('image-files').addEventListener('change', previewImages);
    
    // Back to albums
    document.getElementById('back-to-albums').addEventListener('click', () => {
        showAlbumsView();
    });
    
    // Delete album
    document.getElementById('delete-album-btn').addEventListener('click', deleteCurrentAlbum);
    
    // Edit album form
    document.getElementById('edit-album-form').addEventListener('submit', updateAlbum);
    
    // Download photo
    document.getElementById('download-photo-btn').addEventListener('click', downloadCurrentPhoto);
    
    // Delete photo
    document.getElementById('delete-photo-btn').addEventListener('click', deleteCurrentPhoto);
}

// Load albums from backend
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

// Render albums grid
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

// Create album card element
function createAlbumCard(album) {
    const card = document.createElement('div');
    card.className = 'album-card';
    card.dataset.albumId = album.id;
    
    const coverImage = album.cover_image || getRandomAlbumCover();
    const photoCount = album.photo_count || 0;
    const createdDate = new Date(album.created_at).toLocaleDateString();
    
    card.innerHTML = `
        <div class="album-cover">
            ${coverImage ? 
                `<img src="${coverImage}" alt="${album.name}">` :
                `<div class="album-cover-placeholder">
                    <i class="bi bi-images"></i>
                </div>`
            }
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

// Get random album cover gradient
function getRandomAlbumCover() {
    const gradients = [
        'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
    ];
    return gradients[Math.floor(Math.random() * gradients.length)];
}

// Open album
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
        } else {
            console.error('Failed to load album:', data.error);
        }
    } catch (error) {
        console.error('Error opening album:', error);
    }
}

// Render album view
function renderAlbumView() {
    document.getElementById('albums-container').parentElement.style.display = 'none';
    document.getElementById('selected-album-view').style.display = 'block';
    
    document.getElementById('selected-album-title').textContent = currentAlbum.name;
    document.getElementById('album-photo-count').textContent = currentAlbum.photos?.length || 0;
    document.getElementById('album-created-date').textContent = new Date(currentAlbum.created_at).toLocaleDateString();
    document.getElementById('album-last-updated').textContent = new Date(currentAlbum.updated_at).toLocaleDateString();
    
    renderPhotos(currentAlbum.photos || []);
}

// Render photos grid
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

// Create photo card element
function createPhotoCard(photo) {
    const card = document.createElement('div');
    card.className = 'photo-card';
    card.dataset.photoId = photo.id;
    
    const uploadDate = new Date(photo.uploaded_at).toLocaleDateString();
    const fileSize = formatFileSize(photo.file_size);
    
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
        card.querySelector('.photo-checkbox').addEventListener('change', (e) => {
            togglePhotoSelection(photo.id, e.target.checked);
        });
    }
    
    return card;
}

// Open photo detail
function openPhotoDetail(photo) {
    document.getElementById('detail-photo-image').src = photo.image_path;
    document.getElementById('detail-photo-filename').textContent = photo.filename;
    document.getElementById('detail-photo-date').textContent = new Date(photo.uploaded_at).toLocaleString();
    document.getElementById('detail-photo-size').textContent = formatFileSize(photo.file_size);
    document.getElementById('detail-photo-style').textContent = photo.style || 'Original';
    
    // Store current photo ID for delete/download
    document.getElementById('detail-photo-image').dataset.photoId = photo.id;
    document.getElementById('detail-photo-image').dataset.photoPath = photo.image_path;
    
    const modal = new bootstrap.Modal(document.getElementById('photoDetailModal'));
    modal.show();
}

// Toggle photo selection
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
            selectedPhotos.push(photoId);
        } else {
            selectedPhotos = selectedPhotos.filter(id => id !== photoId);
        }
    }
    
    // Update selection bar
    updateSelectionBar();
}

// Update selection bar
function updateSelectionBar() {
    let bar = document.querySelector('.selection-bar');
    
    if (selectedPhotos.length > 0) {
        if (!bar) {
            bar = document.createElement('div');
            bar.className = 'selection-bar';
            document.getElementById('selected-album-view').insertBefore(bar, document.getElementById('photos-container'));
        }
        
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
    } else if (bar) {
        bar.remove();
        exitSelectionMode();
    }
}

// Enter selection mode
function enterSelectionMode() {
    isSelectionMode = true;
    renderPhotos(currentAlbum.photos || []);
}

// Exit selection mode
function exitSelectionMode() {
    isSelectionMode = false;
    selectedPhotos = [];
    renderPhotos(currentAlbum.photos || []);
    
    const bar = document.querySelector('.selection-bar');
    if (bar) bar.remove();
}

// Show albums view
function showAlbumsView() {
    document.getElementById('albums-container').parentElement.style.display = 'block';
    document.getElementById('selected-album-view').style.display = 'none';
    currentAlbum = null;
    exitSelectionMode();
}

// Create album
async function createAlbum(e) {
    e.preventDefault();
    
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
        const response = await fetch('/api/albums', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            bootstrap.Modal.getInstance(document.getElementById('createAlbumModal')).hide();
            loadAlbums();
            document.getElementById('create-album-form').reset();
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

// Upload images
async function uploadImages(e) {
    e.preventDefault();
    
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
        const response = await fetch('/api/albums/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            bootstrap.Modal.getInstance(document.getElementById('uploadImageModal')).hide();
            openAlbum(currentAlbum.id);
            document.getElementById('upload-images-form').reset();
            document.getElementById('image-preview-container').innerHTML = '';
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

// Preview images before upload
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

// Edit album
async function editAlbum(albumId) {
    const album = albums.find(a => a.id === albumId);
    if (!album) return;
    
    document.getElementById('edit-album-id').value = album.id;
    document.getElementById('edit-album-name').value = album.name;
    document.getElementById('edit-album-description').value = album.description || '';
    
    const modal = new bootstrap.Modal(document.getElementById('editAlbumModal'));
    modal.show();
}

// Update album
async function updateAlbum(e) {
    e.preventDefault();
    
    const albumId = document.getElementById('edit-album-id').value;
    
    const response = await fetch(`/api/albums/${albumId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: document.getElementById('edit-album-name').value,
            description: document.getElementById('edit-album-description').value
        })
    });
    
    const data = await response.json();
    
    if (data.code === 200) {
        bootstrap.Modal.getInstance(document.getElementById('editAlbumModal')).hide();
        loadAlbums();
        if (currentAlbum && currentAlbum.id === albumId) {
            openAlbum(albumId);
        }
    } else {
        alert(data.error || 'Failed to update album');
    }
}

// Delete album
async function deleteAlbum(albumId) {
    if (!confirm('Are you sure you want to delete this album? All photos will be deleted.')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/albums/${albumId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            if (currentAlbum && currentAlbum.id === albumId) {
                showAlbumsView();
            }
            loadAlbums();
        } else {
            alert(data.error || 'Failed to delete album');
        }
    } catch (error) {
        console.error('Error deleting album:', error);
        alert('Failed to delete album');
    }
}

// Delete current album
function deleteCurrentAlbum() {
    if (currentAlbum) {
        deleteAlbum(currentAlbum.id);
    }
}

// Delete photo
async function deletePhoto(photoId) {
    if (!confirm('Are you sure you want to delete this photo?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/photos/${photoId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            bootstrap.Modal.getInstance(document.getElementById('photoDetailModal')).hide();
            openAlbum(currentAlbum.id);
        } else {
            alert(data.error || 'Failed to delete photo');
        }
    } catch (error) {
        console.error('Error deleting photo:', error);
        alert('Failed to delete photo');
    }
}

// Delete current photo
function deleteCurrentPhoto() {
    const photoId = document.getElementById('detail-photo-image').dataset.photoId;
    if (photoId) {
        deletePhoto(photoId);
    }
}

// Delete selected photos
async function deleteSelected() {
    if (selectedPhotos.length === 0) return;
    
    if (!confirm(`Delete ${selectedPhotos.length} selected photos?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/photos/batch-delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ photo_ids: selectedPhotos })
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            exitSelectionMode();
            openAlbum(currentAlbum.id);
        } else {
            alert(data.error || 'Failed to delete photos');
        }
    } catch (error) {
        console.error('Error deleting photos:', error);
        alert('Failed to delete photos');
    }
}

// Download photo
function downloadPhoto(photoPath, filename) {
    const link = document.createElement('a');
    link.href = photoPath;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Download current photo
function downloadCurrentPhoto() {
    const photoPath = document.getElementById('detail-photo-image').dataset.photoPath;
    const filename = document.getElementById('detail-photo-filename').textContent;
    if (photoPath && filename) {
        downloadPhoto(photoPath, filename);
    }
}

// Download selected photos
async function downloadSelected() {
    if (selectedPhotos.length === 0) return;
    
    // For multiple downloads, create a zip file
    try {
        const response = await fetch('/api/photos/batch-download', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ photo_ids: selectedPhotos })
        });
        
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

// Show empty state
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

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}