// DOM Elements
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const transformBtn = document.getElementById('transform-btn');
const resultSection = document.getElementById('result-section');
const originalImg = document.getElementById('original-img');
const processedImg = document.getElementById('processed-img');
const loadingOverlay = document.getElementById('loading-spinner');
const downloadBtn = document.getElementById('download-btn');
const styleSelect = document.getElementById('style-select');
const fileInfo = document.getElementById('file-info');
const filenameSpan = document.getElementById('filename');
const filesizeSpan = document.getElementById('filesize');
const clearFileBtn = document.getElementById('clear-file');
const styleCards = document.querySelectorAll('.style-card');
const styleSection = document.getElementById('style-section');
const actionButtons = document.getElementById('action-buttons');
const resetBtn = document.getElementById('reset-btn');
const steps = document.querySelectorAll('.step');

// State variables
let selectedFile = null;
let selectedStyle = 'sketch';

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Update step indicator
function updateSteps(stepNumber) {
    steps.forEach((step, index) => {
        if (index < stepNumber) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

// Handle file selection
function handleFileSelection(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showAlert('error', 'Please select a valid image file (JPEG or PNG)');
        return;
    }
    
    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showAlert('error', 'File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Update file info display
    filenameSpan.textContent = file.name;
    filesizeSpan.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('d-none');
    
    // Preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImg.src = e.target.result;
        
        // Show style section and action buttons
        styleSection.style.display = 'block';
        actionButtons.style.display = 'block';
        resultSection.style.display = 'none';
        
        // Enable transform button
        transformBtn.disabled = false;
        
        // Update steps
        updateSteps(2);
        
        // Select first style by default
        selectStyleCard('sketch');
    };
    reader.onerror = function() {
        showAlert('error', 'Failed to read the image file');
    };
    reader.readAsDataURL(file);
}

// Select style card
function selectStyleCard(style) {
    selectedStyle = style;
    styleSelect.value = style;
    
    // Update card selection
    styleCards.forEach(card => {
        if (card.dataset.style === style) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
}

// Clear file selection
function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.add('d-none');
    styleSection.style.display = 'none';
    actionButtons.style.display = 'none';
    resultSection.style.display = 'none';
    transformBtn.disabled = true;
    transformBtn.innerHTML = '<i class="bi bi-magic me-2"></i> Transform Image';
    updateSteps(1);
}

// Show alert
function showAlert(type, message) {
    const alertType = type === 'error' ? 'danger' : 'success';
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${alertType} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    alertDiv.style.zIndex = '1050';
    alertDiv.style.minWidth = '300px';
    alertDiv.innerHTML = `
        <i class="bi ${type === 'error' ? 'bi-exclamation-triangle' : 'bi-check-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Event Listeners

// Click upload area
dropZone.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

// Style card selection
styleCards.forEach(card => {
    card.addEventListener('click', () => {
        selectStyleCard(card.dataset.style);
    });
});

// Transform button click
transformBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI updates
    transformBtn.disabled = true;
    transformBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Processing...';
    processedImg.src = '';
    loadingOverlay.style.display = 'flex';
    resultSection.style.display = 'block';
    downloadBtn.style.display = 'none';
    
    // Update steps
    updateSteps(3);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('style', selectedStyle);

    try {
        // Send request to server
        const response = await fetch('/api/transform', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Success - update processed image
            processedImg.src = data.processedUrl;
            downloadBtn.href = data.processedUrl;
            
            // Set download filename
            const timestamp = new Date().getTime();
            downloadBtn.download = `styletrans-${selectedStyle}-${timestamp}.jpg`;
            
            downloadBtn.style.display = 'inline-block';
            
            // Show success message
            showAlert('success', 'Image transformed successfully!');
        } else {
            // Error handling
            showAlert('error', 'Error: ' + (data.error || 'Processing failed'));
        }
    } catch (error) {
        console.error('Transform error:', error);
        showAlert('error', 'Network error. Please check your connection and try again.');
    } finally {
        // Reset UI
        loadingOverlay.style.display = 'none';
        transformBtn.disabled = false;
        transformBtn.innerHTML = '<i class="bi bi-magic me-2"></i> Transform Image';
    }
});

// ==================== Authentication UI Only ====================

// 1. When user clicks "Sign In" button, show auth modal
const loginBtn = document.querySelector('.btn-login');
if (loginBtn) {
    loginBtn.addEventListener('click', () => {
        const authModal = document.getElementById('authModal');
        if (authModal) {
            const modal = new bootstrap.Modal(authModal);
            modal.show();
        }
    });
}

// 2. Basic form validation feedback (UI only)
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        alert('Login feature will be implemented with backend');
    });
}

const registerForm = document.getElementById('registerForm');
if (registerForm) {
    registerForm.addEventListener('submit', (e) => {
        e.preventDefault();
        alert('Registration feature will be implemented with backend');
    });
}

// 3. Show loading state on form submission (visual feedback only)
function showFormLoading(form, isLoading) {
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
        if (isLoading) {
            submitBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Processing...';
            submitBtn.disabled = true;
        } else {
            // Reset based on form type
            if (form.id === 'loginForm') {
                submitBtn.innerHTML = '<i class="bi bi-box-arrow-in-right me-2"></i> Sign In';
            } else {
                submitBtn.innerHTML = '<i class="bi bi-person-plus me-2"></i> Create Account';
            }
            submitBtn.disabled = false;
        }
    }
}

// Reset button
resetBtn.addEventListener('click', clearFileSelection);

// Clear file button
clearFileBtn.addEventListener('click', clearFileSelection);

// Initialize
updateSteps(1);
