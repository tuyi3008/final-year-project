// ====================== DOM Elements ======================
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const transformBtn = document.getElementById('transform-btn');
const resultSection = document.getElementById('result-section');
const originalImg = document.getElementById('original-img');
const processedImg = document.getElementById('processed-img');
const loadingOverlay = document.getElementById('loading-spinner');
const downloadBtn = document.getElementById('download-btn');
const publishBtn = document.getElementById('publish-btn');
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

const aspectRatioSelect = document.getElementById('aspect-ratio');

// ====================== State Variables ======================
let selectedFile = null;
let selectedStyle = 'sketch';
let currentProcessedImage = null; // Store the processed image base64 for publishing

// ====================== Helper Functions ======================

// Format bytes into human-readable format
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

// Display alert message
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
    setTimeout(() => {
        if (alertDiv.parentNode) alertDiv.remove();
    }, 5000);
}

function handleFileSelection(file) {
    console.log('File selected:', file.name);
    
    // Validate file type and size
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showAlert('error', 'Please select a valid image file (JPEG or PNG)');
        return;
    }
    
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showAlert('error', 'File size must be less than 10MB');
        return;
    }

    selectedFile = file;

    // display file info
    fileInfo.classList.remove('d-none');
    filenameSpan.textContent = file.name;
    filesizeSpan.textContent = formatFileSize(file.size);
    
    // fetch and display the image preview
    const uploadPreview = document.getElementById('upload-preview');
    const uploadIcon = document.getElementById('upload-icon');
    const uploadText = document.getElementById('upload-text');
    const uploadSubtext = document.getElementById('upload-subtext');
    const browseBtn = document.getElementById('browse-btn');
    
    // fetch thumbnail elements
    const thumbnailImg = document.getElementById('file-thumbnail-img');
    const placeholder = document.getElementById('file-thumbnail-placeholder');
    
    // read the file and display preview
    const reader = new FileReader();
    reader.onload = function(e) {
        console.log('FileReader loaded');
        
        // display the uploaded image in the upload zone
        if (uploadPreview) {
            uploadPreview.src = e.target.result;
            uploadPreview.style.display = 'block';
        }
        
        // hide the original upload prompts
        if (uploadIcon) uploadIcon.style.display = 'none';
        if (uploadText) uploadText.style.display = 'none';
        if (uploadSubtext) uploadSubtext.style.display = 'none';
        if (browseBtn) browseBtn.style.display = 'none';
        
        // display thumbnail in file info section
        if (thumbnailImg) {
            thumbnailImg.src = e.target.result;
            thumbnailImg.style.display = 'block';
        }
        if (placeholder) {
            placeholder.style.display = 'none';
        }
        
        // display original image in result section
        if (originalImg) {
            originalImg.src = e.target.result;
        }
        
        // show style options and action buttons
        styleSection.style.display = 'block';
        actionButtons.style.display = 'block';
        resultSection.style.display = 'none';
        transformBtn.disabled = false;
        updateSteps(2);
        selectStyleCard('sketch');
    };
    
    reader.readAsDataURL(file);
}

// Select style card
function selectStyleCard(style) {
    selectedStyle = style;
    styleSelect.value = style;

    styleCards.forEach(card => {
        if (card.dataset.style === style) card.classList.add('selected');
        else card.classList.remove('selected');
    });
}

function clearFileSelection() {
    selectedFile = null;
    currentProcessedImage = null;
    fileInput.value = '';
    
    // Reset upload zone
    const uploadPreview = document.getElementById('upload-preview');
    const uploadIcon = document.getElementById('upload-icon');
    const uploadText = document.getElementById('upload-text');
    const uploadSubtext = document.getElementById('upload-subtext');
    const browseBtn = document.getElementById('browse-btn');
    
    if (uploadPreview) {
        uploadPreview.src = '';
        uploadPreview.style.display = 'none';
    }
    if (uploadIcon) uploadIcon.style.display = 'block';
    if (uploadText) uploadText.style.display = 'block';
    if (uploadSubtext) uploadSubtext.style.display = 'block';
    if (browseBtn) browseBtn.style.display = 'inline-block';
    
    // Reset thumbnail
    const thumbnailImg = document.getElementById('file-thumbnail-img');
    const placeholder = document.getElementById('file-thumbnail-placeholder');
    if (thumbnailImg) {
        thumbnailImg.src = '';
        thumbnailImg.style.display = 'none';
    }
    if (placeholder) {
        placeholder.style.display = 'flex';
    }
    
    fileInfo.classList.add('d-none');
    styleSection.style.display = 'none';
    actionButtons.style.display = 'none';
    resultSection.style.display = 'none';
    transformBtn.disabled = true;
    transformBtn.innerHTML = '<i class="bi bi-magic me-2"></i> Transform Image';
    
    // Hide action buttons
    if (downloadBtn) downloadBtn.style.display = 'none';
    if (publishBtn) {
        publishBtn.style.display = 'none';
        publishBtn.disabled = false;
        publishBtn.innerHTML = '<i class="bi bi-share me-2"></i> Publish to Gallery';
    }
    
    updateSteps(1);
}

// ====================== Publish Functionality ======================
async function handlePublish() {
    if (!currentProcessedImage || !selectedStyle) {
        showAlert('error', 'No image to publish');
        return;
    }
    
    // Check if user is logged in
    const token = localStorage.getItem('token');
    if (!token) {
        showAlert('error', 'Please login to publish to gallery');
        if (window.auth && window.auth.showLoginModal) {
            window.auth.showLoginModal();
        }
        return;
    }
    
    try {
        // Show loading state
        publishBtn.disabled = true;
        publishBtn.innerHTML = '<i class="bi bi-hourglass me-2"></i> Publishing...';
        
        const response = await fetch('http://127.0.0.1:8000/gallery/publish', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_base64: currentProcessedImage,
                style: selectedStyle,
                title: `My ${selectedStyle} creation`,
                description: `Created with StyleTrans AI on ${new Date().toLocaleDateString()}`
            })
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            showAlert('success', '✨ Published to gallery successfully!');
            publishBtn.innerHTML = '<i class="bi bi-check-circle me-2"></i> Published!';
            
            // Reset button text after 3 seconds
            setTimeout(() => {
                if (publishBtn && !publishBtn.disabled) {
                    publishBtn.innerHTML = '<i class="bi bi-share me-2"></i> Publish to Gallery';
                }
            }, 3000);
        } else {
            showAlert('error', data.error || 'Failed to publish');
            publishBtn.innerHTML = '<i class="bi bi-share me-2"></i> Publish to Gallery';
        }
    } catch (error) {
        console.error('Publish error:', error);
        showAlert('error', 'Network error. Please try again.');
        publishBtn.innerHTML = '<i class="bi bi-share me-2"></i> Publish to Gallery';
    } finally {
        publishBtn.disabled = false;
    }
}

// ====================== Event Listeners ======================

// Click upload zone to trigger file input
dropZone.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0]);
});

// Style card selection
styleCards.forEach(card => {
    card.addEventListener('click', () => selectStyleCard(card.dataset.style));
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
    if (publishBtn) publishBtn.style.display = 'none';
    updateSteps(3);

    // Prepare form data
    const formData = new FormData();
    formData.append('content', selectedFile);
    formData.append('style', selectedStyle);

    const aspectRatio = aspectRatioSelect ? aspectRatioSelect.value : 'auto';
    formData.append('aspect_ratio', aspectRatio);
    console.log('Selected aspect ratio:', aspectRatio);

    try {
        const token = localStorage.getItem('token');
        console.log('Token from localStorage:', token ? token.substring(0, 20) + '...' : 'No token');

        const response = await fetch('http://127.0.0.1:8000/stylize/', {
            method: 'POST',
            headers: token ? { 'Authorization': 'Bearer ' + token } : {},
            body: formData
        });

        console.log('Response status:', response.status);

        // if (response.status === 401) {
        //     showAlert('error', 'Please login to transform images');
        //     if (window.auth && window.auth.showLoginModal) {
        //         window.auth.showLoginModal();
        //     }
        //     transformBtn.disabled = false;
        //     transformBtn.innerHTML = '<i class="bi bi-magic me-2"></i> Transform Image';
        //     loadingOverlay.style.display = 'none';
        //     return;
        // }

        const data = await response.json();

        if (data.image_base64) {
            const base64Data = `data:image/png;base64,${data.image_base64}`;
            processedImg.src = base64Data;
            downloadBtn.href = base64Data;
            
            // Store the raw base64 for publishing (without the data:image prefix)
            currentProcessedImage = data.image_base64;

            const timestamp = new Date().getTime();
            downloadBtn.download = `styletrans-${selectedStyle}-${timestamp}.png`;
            downloadBtn.style.display = 'inline-block';
            
            // Show publish button
            if (publishBtn) {
                publishBtn.style.display = 'inline-block';
                publishBtn.innerHTML = '<i class="bi bi-share me-2"></i> Publish to Gallery';
            }
            
            showAlert('success', 'Image transformed successfully!');

            if (data.crop_info) {
                console.log('Cropping info:', data.crop_info);
            }
        } else {
            showAlert('error', 'Processing failed');
        }

    } catch (error) {
        console.error('Transform error:', error);
        showAlert('error', 'Network error. Please check your connection.');
    } finally {
        loadingOverlay.style.display = 'none';
        transformBtn.disabled = false;
        transformBtn.innerHTML = '<i class="bi bi-magic me-2"></i> Transform Image';
    }
});

// Publish button click
if (publishBtn) {
    publishBtn.addEventListener('click', handlePublish);
}

// Reset button
resetBtn.addEventListener('click', clearFileSelection);

// Clear file button
clearFileBtn.addEventListener('click', clearFileSelection);

// Initialize steps
updateSteps(1);