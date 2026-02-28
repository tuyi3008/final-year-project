// community.js - Community Hub Functionality

let currentFilter = 'all';
let currentSort = 'popular';
let submissionsData = [];
let weeklyChallenge = null;
let page = 1;
const itemsPerPage = 8;
let isLoading = false;
let hasMore = true;

// Load weekly challenge
async function loadWeeklyChallenge() {
    try {
        const response = await fetch('/api/challenge/current');
        const data = await response.json();
        
        if (data.code === 200 && data.challenge) {
            weeklyChallenge = data.challenge;
            renderWeeklyChallenge();
        }
    } catch (error) {
        console.error('Error loading challenge:', error);
    }
}

// Render weekly challenge
function renderWeeklyChallenge() {
    const challengeEl = document.getElementById('weekly-challenge');
    
    if (!challengeEl || !weeklyChallenge) return;
    
    const startDate = new Date(weeklyChallenge.start_date).toLocaleDateString();
    const endDate = new Date(weeklyChallenge.end_date).toLocaleDateString();
    
    document.getElementById('challenge-dates').textContent = 
        `${startDate} - ${endDate}`;
    
    challengeEl.innerHTML = `
        <div class="challenge-header">
            <div class="challenge-icon">
                <i class="bi bi-trophy-fill"></i>
            </div>
            <div class="challenge-info">
                <h3>${weeklyChallenge.theme}</h3>
                <p class="challenge-dates">
                    <i class="bi bi-calendar3 me-1"></i>
                    ${startDate} - ${endDate}
                </p>
            </div>
        </div>
        <p class="challenge-description">${weeklyChallenge.description}</p>
        <div class="challenge-stats">
            <div class="stat-item">
                <span class="stat-value">${weeklyChallenge.submissions || 0}</span>
                <span class="stat-label">Submissions</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${weeklyChallenge.participants || 0}</span>
                <span class="stat-label">Participants</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${weeklyChallenge.days_left || 0}</span>
                <span class="stat-label">Days Left</span>
            </div>
        </div>
        <div class="challenge-actions">
            <button class="btn-challenge" onclick="viewChallengeDetails()">
                <i class="bi bi-info-circle"></i>
                View Details
            </button>
            <button class="btn-outline-challenge" onclick="showJoinChallengeModal()">
                <i class="bi bi-upload"></i>
                Submit Your Art
            </button>
        </div>
        ${weeklyChallenge.winners ? renderWinners(weeklyChallenge.winners) : ''}
    `;
}

// Render winners section
function renderWinners(winners) {
    let winnersHtml = '<div class="winners-section"><h5 class="winners-title">🏆 Top Winners</h5><div class="winners-list">';
    
    winners.forEach((winner, index) => {
        winnersHtml += `
            <div class="winner-item">
                <div class="winner-rank">${index + 1}</div>
                <div class="winner-info">
                    <div class="winner-name">${winner.username}</div>
                    <div class="winner-likes">❤️ ${winner.likes} likes</div>
                </div>
            </div>
        `;
    });
    
    winnersHtml += '</div></div>';
    return winnersHtml;
}

// Load trending artworks (most liked from gallery)
async function loadTrendingArtworks() {
    try {
        const response = await fetch('/gallery/images');
        const data = await response.json();
        
        if (data.code === 200 && data.images) {
            const trending = data.images
                .sort((a, b) => (b.likes || 0) - (a.likes || 0))
                .slice(0, 4);
            
            renderTrendingArtworks(trending);
        }
    } catch (error) {
        console.error('Error loading trending artworks:', error);
    }
}

// Render trending artworks
function renderTrendingArtworks(artworks) {
    const container = document.getElementById('trending-container');
    
    if (!container) return;
    
    if (artworks.length === 0) {
        container.innerHTML = `
            <div class="col-12">
                <div class="empty-state">
                    <i class="bi bi-images"></i>
                    <h4>No trending artworks yet</h4>
                    <p class="text-muted">Be the first to create something amazing!</p>
                </div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    artworks.forEach((artwork, index) => {
        const item = document.createElement('div');
        item.className = 'trending-item';
        item.onclick = () => viewSubmission(artwork);
        
        const date = artwork.created_at ? new Date(artwork.created_at).toLocaleDateString() : 'Recent';
        
        item.innerHTML = `
            <div class="trending-rank">#${index + 1}</div>
            <div class="trending-likes">
                <i class="bi bi-heart-fill"></i> ${artwork.likes || 0}
            </div>
            <img src="/${artwork.image_path}" alt="${artwork.style}">
            <div class="trending-overlay">
                <h6 class="mb-1">${artwork.username || 'Anonymous'}</h6>
                <p class="small mb-0">
                    <i class="bi bi-calendar3 me-1"></i>${date}
                </p>
                <span class="badge bg-primary mt-1">${artwork.style}</span>
            </div>
        `;
        
        container.appendChild(item);
    });
}

// Load challenge submissions
async function loadSubmissions(reset = false) {
    if (isLoading) return;
    
    if (reset) {
        page = 1;
        hasMore = true;
        submissionsData = [];
    }
    
    if (!hasMore) return;
    
    isLoading = true;
    const loadingMore = document.getElementById('loading-more');
    if (loadingMore) loadingMore.style.display = 'block';
    
    try {
        const url = `/api/challenge/submissions?page=${page}&limit=${itemsPerPage}&style=${currentFilter}&sort=${currentSort}`;
        console.log('📡 Fetching:', url);
        
        const response = await fetch(url);
        const data = await response.json();

        console.log('📦 API response:', data);
        
        if (data.code === 200) {
            const validSubmissions = (data.submissions || []).filter(item => item != null);
            console.log('✅ Valid submissions:', validSubmissions.length);
            
            if (reset) {
                submissionsData = validSubmissions;
            } else {
                submissionsData = [...submissionsData, ...validSubmissions];
            }
            
            renderSubmissions();
            hasMore = data.has_more || false;
        } else {
            console.error('❌ API error:', data.error);
        }
    } catch (error) {
        console.error('❌ Fetch error:', error);
    } finally {
        isLoading = false;
        if (loadingMore) loadingMore.style.display = 'none';
        
        const loadMoreBtn = document.getElementById('load-more-submissions');
        if (loadMoreBtn) {
            loadMoreBtn.style.display = hasMore ? 'block' : 'none';
        }
    }
}

// Sort submissions
function sortSubmissions(submissions, sortBy) {
    switch(sortBy) {
        case 'popular':
            return [...submissions].sort((a, b) => (b.likes || 0) - (a.likes || 0));
        case 'latest':
            return [...submissions].sort((a, b) => 
                new Date(b.created_at || 0) - new Date(a.created_at || 0));
        default:
            return submissions;
    }
}

// Render submissions
function renderSubmissions() {
    const container = document.getElementById('submissions-container');
    
    if (!container) return;
    
    if (!submissionsData || submissionsData.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="bi bi-images"></i>
                <h4>No submissions yet</h4>
                <p class="text-muted">Be the first to join the challenge!</p>
                <button class="btn btn-gradient" onclick="showJoinChallengeModal()">
                    <i class="bi bi-upload me-2"></i>
                    Join Challenge
                </button>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    submissionsData.forEach(item => {
        const submissionEl = document.createElement('div');
        submissionEl.className = 'submission-item';
        
        const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent';
        const style = item.style || 'Unknown';
        const username = item.username || 'Anonymous';
        const likes = item.likes || 0;
        const isLiked = likes > 0; // 这里需要根据实际情况判断是否已点赞
        
        let imagePath = item.image_path || '';

        if (imagePath && !imagePath.startsWith('http')) {
            if (!imagePath.startsWith('/')) {
                imagePath = '/' + imagePath;
            }
            if (!imagePath.includes('/uploads/')) {
                imagePath = '/uploads/album_photos/' + imagePath.split('/').pop();
            }
        }

        const imgSrc = imagePath ? `http://localhost:8000${imagePath}` : 'https://via.placeholder.com/300?text=No+Image';
        
        submissionEl.innerHTML = `
            <div class="submission-image">
                <img src="${imgSrc}" alt="${style}" 
                     onerror="this.src='https://via.placeholder.com/300?text=No+Image'"
                     style="width:100%; height:100%; object-fit:cover;">
                ${weeklyChallenge ? `
                    <div class="challenge-badge">
                        <i class="bi bi-trophy-fill"></i>
                        ${weeklyChallenge.theme}
                    </div>
                ` : ''}
            </div>
            <div class="submission-info">
                <div class="submission-user">
                    <i class="bi bi-person-circle"></i>
                    <span>${username}</span>
                </div>
                <div class="submission-stats">
                    <button class="btn-like ${isLiked ? 'liked' : ''}" 
                            onclick="likeSubmission('${item._id}')"
                            data-likes="${likes}">
                        <i class="bi ${isLiked ? 'bi-heart-fill' : 'bi-heart'}"></i>
                        <span class="likes-count">${likes}</span>
                    </button>
                    <span class="submission-style">${style}</span>
                </div>
                <div class="small text-muted mt-2">
                    <i class="bi bi-calendar3 me-1"></i>${date}
                </div>
            </div>
        `;
        
        submissionEl.addEventListener('click', function(e) {
            // 防止点击按钮时触发
            if (e.target.closest('.btn-like')) return;
            viewSubmission(item);
        });
        
        container.appendChild(submissionEl);
    });
}

// View submission details
function viewSubmission(item) {
    console.log('Viewing submission:', item);
    console.log('Submission ID:', item._id);
    const modalContent = document.getElementById('submission-detail-content');
    const modalElement = document.getElementById('submissionModal');
    
    if (!modalContent || !modalElement) {
        console.error('Modal elements not found');
        return;
    }
    
    const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent';
    const isLiked = item.likes > 0; // 或者根据实际点赞状态判断
    
    let imagePath = item.image_path || '';
    if (imagePath && !imagePath.startsWith('http')) {
        if (!imagePath.startsWith('/')) {
            imagePath = '/' + imagePath;
        }
        if (!imagePath.includes('/uploads/')) {
            imagePath = '/uploads/album_photos/' + imagePath.split('/').pop();
        }
        imagePath = `http://localhost:8000${imagePath}`;
    }
    
    modalContent.innerHTML = `
        <div class="row g-0">
            <div class="col-md-7">
                <img src="${imagePath || 'https://via.placeholder.com/500?text=No+Image'}" 
                     class="img-fluid w-100" alt="${item.style}" 
                     style="max-height: 500px; object-fit: cover;"
                     onerror="this.src='https://via.placeholder.com/500?text=No+Image'">
            </div>
            <div class="col-md-5 p-4">
                <div class="d-flex align-items-center mb-3">
                    <i class="bi bi-person-circle fs-1 me-3" style="color: #667eea;"></i>
                    <div>
                        <h5 class="mb-1">${item.username || 'Anonymous'}</h5>
                        <p class="text-muted mb-0">
                            <i class="bi bi-calendar3 me-1"></i>${date}
                        </p>
                    </div>
                </div>
                
                <div class="mb-3">
                    <span class="badge bg-primary me-2">${item.style}</span>
                    ${weeklyChallenge ? `
                        <span class="badge bg-warning">
                            <i class="bi bi-trophy-fill"></i> ${weeklyChallenge.theme}
                        </span>
                    ` : ''}
                </div>
                
                <div class="d-flex gap-4 mb-4">
                    <div class="text-center">
                        <h4 class="mb-0">${item.likes || 0}</h4>
                        <small class="text-muted">Likes</small>
                    </div>
                    <div class="text-center">
                        <h4 class="mb-0">${item.views || 0}</h4>
                        <small class="text-muted">Views</small>
                    </div>
                </div>
                
                <div class="d-flex gap-2">
                    <button class="btn btn-outline-light flex-grow-1 btn-like ${isLiked ? 'liked' : ''}" 
                            onclick="likeSubmission('${item._id}')"
                            data-likes="${item.likes || 0}">
                        <i class="bi ${isLiked ? 'bi-heart-fill' : 'bi-heart'}"></i>
                        <span class="likes-count">${item.likes || 0}</span>
                    </button>
                    <button class="btn btn-outline-light flex-grow-1" onclick="shareSubmission('${item._id}')">
                        <i class="bi bi-share"></i> Share
                    </button>
                </div>
            </div>
        </div>
    `;
    
    try {
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
    } catch (error) {
        console.error('Error showing modal:', error);
    }
}

// View challenge details
function viewChallengeDetails() {
    if (!weeklyChallenge) return;
    
    const modalContent = document.getElementById('challenge-detail-content');
    
    modalContent.innerHTML = `
        <div class="p-4">
            <h4 class="mb-3">${weeklyChallenge.theme}</h4>
            <p class="mb-4">${weeklyChallenge.description}</p>
            
            <h5 class="mb-3">📋 Rules</h5>
            <ul class="mb-4">
                <li>Use any of our AI styles (Sketch, Anime, or Ink Wash)</li>
                <li>Submit your original creation</li>
                <li>One submission per participant</li>
                <li>Vote for your favorite submissions</li>
            </ul>
            
            <h5 class="mb-3">🏆 Prizes</h5>
            <ul class="mb-4">
                <li>🥇 1st Place: Featured on homepage + 500 points</li>
                <li>🥈 2nd Place: 300 points</li>
                <li>🥉 3rd Place: 100 points</li>
            </ul>
            
            <div class="d-grid">
                <button class="btn btn-gradient" onclick="showJoinChallengeModal()">
                    <i class="bi bi-upload me-2"></i>
                    Join Challenge
                </button>
            </div>
        </div>
    `;
    
    const modal = new bootstrap.Modal(document.getElementById('challengeModal'));
    modal.show();
}

// Show upload option
function showUploadOption() {
    document.getElementById('source-options').style.display = 'none';
    document.getElementById('no-transformed-image').style.display = 'none';
    document.getElementById('upload-content').style.display = 'block';
    
    document.getElementById('challenge-file-input').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('upload-preview').style.display = 'block';
                document.getElementById('upload-preview img').src = e.target.result;
                document.getElementById('upload-form').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
}

// Select from album
function selectFromAlbum() {
    if (weeklyChallenge) {
        sessionStorage.setItem('pendingChallenge', JSON.stringify(weeklyChallenge));
    }
    window.location.href = '/static/photos.html?select=challenge';
}

// Reset join modal
function resetJoinModal() {
    document.getElementById('source-options').style.display = 'flex';
    document.getElementById('no-transformed-image').style.display = 'block';
    document.getElementById('upload-content').style.display = 'none';
    document.getElementById('upload-preview').style.display = 'none';
    document.getElementById('upload-form').style.display = 'none';
    document.getElementById('challenge-file-input').value = '';
}

// Process and submit uploaded file
async function processAndSubmit() {
    const file = document.getElementById('challenge-file-input').files[0];
    const style = document.getElementById('upload-style').value;
    const description = document.getElementById('upload-description').value;
    const token = localStorage.getItem('token');
    
    if (!file) {
        alert('Please select a file');
        return;
    }
    
    if (!description.trim()) {
        alert('Please add a description');
        return;
    }
    
    const submitBtn = document.querySelector('#upload-form .btn-gradient');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    
    try {
        const formData = new FormData();
        formData.append('content', file);
        formData.append('style', style);
        
        const transformResponse = await fetch('http://localhost:8000/stylize/', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token
            },
            body: formData
        });
        
        const transformData = await transformResponse.json();
        
        if (!transformResponse.ok) {
            throw new Error(transformData.error || 'Transformation failed');
        }
        
        const submitResponse = await fetch('/api/challenge/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token
            },
            body: JSON.stringify({
                image_path: transformData.image_path,
                style: style,
                description: description,
                challenge_id: weeklyChallenge?.id
            })
        });
        
        const submitData = await submitResponse.json();
        
        if (submitData.code === 200) {
            alert('Successfully submitted to challenge!');
            
            const modal = bootstrap.Modal.getInstance(document.getElementById('joinChallengeModal'));
            modal.hide();
            
            resetJoinModal();
            loadSubmissions(true);
        } else {
            alert(submitData.error || 'Failed to submit');
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to process image. Please try again.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="bi bi-cloud-upload me-2"></i>Process & Submit';
    }
}

// Show join challenge modal
async function showJoinChallengeModal() {
    const token = localStorage.getItem('token');
    
    if (!token) {
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal('Please login to join the challenge');
        }
        return;
    }
    
    resetJoinModal();
    
    const modal = new bootstrap.Modal(document.getElementById('joinChallengeModal'));
    modal.show();
}

// Filter submissions
function filterSubmissions(filter) {
    currentFilter = filter;
    
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.filter === filter) {
            btn.classList.add('active');
        }
    });
    
    loadSubmissions(true);
}

// Sort submissions
function sortSubmissionsList() {
    const sortSelect = document.getElementById('submissions-sort');
    currentSort = sortSelect.value;
    loadSubmissions(true);
}

// Like submission
async function likeSubmission(id) {
    const token = localStorage.getItem('token');
    
    if (!token) {
        if (window.auth?.showLoginModal) {
            window.auth.showLoginModal('Please login to like submissions');
        }
        return;
    }
    
    // 获取当前点击的按钮
    const clickedButton = event?.target?.closest('button');
    if (!clickedButton) return;
    
    const isLiked = clickedButton.classList.contains('liked');
    const likesSpan = clickedButton.querySelector('.likes-count');
    let currentLikes = parseInt(likesSpan?.textContent || '0');
    
    // Disable button during API call
    clickedButton.disabled = true;
    
    try {
        const response = await fetch(`/api/submission/${id}/like`, {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token
            }
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            // 计算新的点赞数
            const newLikes = data.liked ? currentLikes + 1 : currentLikes - 1;
            
            // 更新本地数据
            const submission = submissionsData.find(s => s._id === id);
            if (submission) {
                submission.likes = newLikes;
            }
            
            // 更新当前点击的按钮
            if (data.liked) {
                clickedButton.classList.add('liked');
                clickedButton.querySelector('i').className = 'bi bi-heart-fill';
            } else {
                clickedButton.classList.remove('liked');
                clickedButton.querySelector('i').className = 'bi bi-heart';
            }
            likesSpan.textContent = newLikes;
            
            // 更新页面上所有相同 ID 的按钮（包括模态框外的其他按钮）
            document.querySelectorAll(`.btn-like[onclick*="'${id}'"]`).forEach(btn => {
                if (btn === clickedButton) return; // 跳过已经更新的按钮
                
                const btnLikesSpan = btn.querySelector('.likes-count');
                if (data.liked) {
                    btn.classList.add('liked');
                    btn.querySelector('i').className = 'bi bi-heart-fill';
                } else {
                    btn.classList.remove('liked');
                    btn.querySelector('i').className = 'bi bi-heart';
                }
                if (btnLikesSpan) btnLikesSpan.textContent = newLikes;
            });
            
            // 如果模态框打开，同时更新模态框里的点赞按钮
            const modalElement = document.getElementById('submissionModal');
            if (modalElement && modalElement.classList.contains('show')) {
                const modalLikeBtn = modalElement.querySelector(`.btn-like[onclick*="'${id}'"]`);
                if (modalLikeBtn) {
                    const modalLikesSpan = modalLikeBtn.querySelector('.likes-count');
                    if (data.liked) {
                        modalLikeBtn.classList.add('liked');
                        modalLikeBtn.querySelector('i').className = 'bi bi-heart-fill';
                    } else {
                        modalLikeBtn.classList.remove('liked');
                        modalLikeBtn.querySelector('i').className = 'bi bi-heart';
                    }
                    if (modalLikesSpan) modalLikesSpan.textContent = newLikes;
                }
                
                // ==== 新增：更新模态框里的 <h4> 标签 ====
                const modalLikesH4 = modalElement.querySelector('.d-flex.gap-4 .text-center:first-child h4');
                if (modalLikesH4) {
                    modalLikesH4.textContent = newLikes;
                }
                // ==== 新增结束 ====
            }
        }
    } catch (error) {
        console.error('Error liking submission:', error);
        alert('Failed to like. Please try again.');
    } finally {
        clickedButton.disabled = false;
    }
}

// Share submission
function shareSubmission(id) {
    const link = `${window.location.origin}/submission/${id}`;
    navigator.clipboard?.writeText(link).then(() => {
        alert('Link copied to clipboard!');
    }).catch(() => {
        alert('Link copied to clipboard!');
    });
}

// View all submissions
function viewAllSubmissions() {
    document.getElementById('submissions-container').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
}

// Load more submissions
function loadMoreSubmissions() {
    if (!isLoading && hasMore) {
        page++;
        loadSubmissions();
    }
}

// Initialize community page
document.addEventListener('DOMContentLoaded', () => {
    loadWeeklyChallenge();
    loadTrendingArtworks();
    loadSubmissions();
    
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            filterSubmissions(e.target.dataset.filter);
        });
    });
    
    const sortSelect = document.getElementById('submissions-sort');
    if (sortSelect) {
        sortSelect.addEventListener('change', sortSubmissionsList);
    }
    
    const loadMoreBtn = document.getElementById('load-more-submissions');
    if (loadMoreBtn) {
        loadMoreBtn.addEventListener('click', loadMoreSubmissions);
    }
    
    const viewAllBtn = document.getElementById('view-all-submissions');
    if (viewAllBtn) {
        viewAllBtn.addEventListener('click', viewAllSubmissions);
    }
    
    const joinModal = document.getElementById('joinChallengeModal');
    if (joinModal) {
        joinModal.addEventListener('hidden.bs.modal', () => {
            resetJoinModal();
        });
    }
});

// Make functions global for onclick handlers
window.showJoinChallengeModal = showJoinChallengeModal;
window.viewChallengeDetails = viewChallengeDetails;
window.showUploadOption = showUploadOption;
window.selectFromAlbum = selectFromAlbum;
window.processAndSubmit = processAndSubmit;
window.likeSubmission = likeSubmission;
window.shareSubmission = shareSubmission;