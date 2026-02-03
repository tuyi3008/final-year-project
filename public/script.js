const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const transformBtn = document.getElementById('transform-btn');
const resultSection = document.getElementById('result-section');
const originalImg = document.getElementById('original-img');
const processedImg = document.getElementById('processed-img');
const loadingOverlay = document.getElementById('loading-spinner');
const downloadBtn = document.getElementById('download-btn');
const styleSelect = document.getElementById('style-select');

let selectedFile = null;

// 1. handle drag and drop
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

// 2. handle drag over
function handleFileSelection(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImg.src = e.target.result;
        // show result section
        resultSection.style.display = 'block';
        processedImg.style.display = 'none'; 
        downloadBtn.style.display = 'none';
        transformBtn.disabled = false;
        transformBtn.textContent = "Transform Image";
    }
    reader.readAsDataURL(file);
}

// 3. handle transform button click
transformBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI updates
    transformBtn.disabled = true;
    processedImg.style.display = 'block'; // show processed image area
    processedImg.src = ''; // clear previous image
    loadingOverlay.style.display = 'flex'; // show loading

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('style', styleSelect.value);

    try {
        // send request to server
        const response = await fetch('/api/transform', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // success
            processedImg.src = data.processedUrl;
            downloadBtn.href = data.processedUrl;
            downloadBtn.style.display = 'inline-block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error(error);
        alert('Something went wrong!');
    } finally {
        loadingOverlay.style.display = 'none';
        transformBtn.disabled = false;
    }
});