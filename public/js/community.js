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
            // Sort by likes (descending) and take top 4
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
    document.getElementById('loading-more').style.display = 'block';
    
    try {
        const response = await fetch('/gallery/images');
        const data = await response.json();
        
        if (data.code === 200 && data.images) {
            // Filter by current filter
            let filtered = data.images;
            if (currentFilter !== 'all') {
                filtered = filtered.filter(img => img.style === currentFilter);
            }
            
            // Sort
            filtered = sortSubmissions(filtered, currentSort);
            
            submissionsData = filtered;
            renderSubmissions();
            
            // Check if there are more items
            hasMore = filtered.length > page * itemsPerPage;
        }
    } catch (error) {
        console.error('Error loading submissions:', error);
    } finally {
        isLoading = false;
        document.getElementById('loading-more').style.display = 'none';
        document.getElementById('load-more-submissions').style.display = 
            hasMore ? 'block' : 'none';
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
    
    if (submissionsData.length === 0) {
        container.innerHTML = `
            <div class="col-12">
                <div class="empty-state">
                    <i class="bi bi-images"></i>
                    <h4>No submissions yet</h4>
                    <p class="text-muted">Be the first to join the challenge!</p>
                    <button class="btn btn-gradient" onclick="showJoinChallengeModal()">
                        <i class="bi bi-upload me-2"></i>
                        Join Challenge
                    </button>
                </div>
            </div>
        `;
        return;
    }
    
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageItems = submissionsData.slice(0, end);
    
    container.innerHTML = '';
    
    pageItems.forEach(item => {
        const submissionEl = document.createElement('div');
        submissionEl.className = 'submission-item';
        submissionEl.onclick = () => viewSubmission(item);
        
        const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent';
        
        submissionEl.innerHTML = `
            <div class="submission-image">
                <img src="/${item.image_path}" alt="${item.style} style">
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
                    <span>${item.username || 'Anonymous'}</span>
                </div>
                <div class="submission-stats">
                    <span class="submission-likes">
                        <i class="bi bi-heart-fill"></i> ${item.likes || 0}
                    </span>
                    <span class="submission-style">${item.style}</span>
                </div>
                <div class="small text-muted mt-2">
                    <i class="bi bi-calendar3 me-1"></i>${date}
                </div>
            </div>
        `;
        
        container.appendChild(submissionEl);
    });
}

// View submission details
function viewSubmission(item) {
    const modalContent = document.getElementById('submission-detail-content');
    
    const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Recent';
    
    modalContent.innerHTML = `
        <div class="row g-0">
            <div class="col-md-7">
                <img src="/${item.image_path}" class="img-fluid w-100" alt="${item.style}" style="max-height: 500px; object-fit: cover;">
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
                    <button class="btn btn-outline-light flex-grow-1" onclick="likeSubmission(${item.id})">
                        <i class="bi bi-heart"></i> Like
                    </button>
                    <button class="btn btn-outline-light flex-grow-1" onclick="shareSubmission(${item.id})">
                        <i class="bi bi-share"></i> Share
                    </button>
                </div>
            </div>
        </div>
    `;
    
    const modal = new bootstrap.Modal(document.getElementById('submissionModal'));
    modal.show();
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

// Show join challenge modal
function showJoinChallengeModal() {
    const lastTransform = localStorage.getItem('lastTransform');
    
    if (lastTransform) {
        const transform = JSON.parse(lastTransform);
        document.getElementById('no-transformed-image').style.display = 'none';
        document.getElementById('join-challenge-form').style.display = 'block';
        document.getElementById('challenge-preview-image').src = transform.image_path;
        document.getElementById('challenge-theme').value = weeklyChallenge?.theme || '';
        document.getElementById('challenge-style').value = transform.style;
    } else {
        document.getElementById('no-transformed-image').style.display = 'block';
        document.getElementById('join-challenge-form').style.display = 'none';
    }
    
    const modal = new bootstrap.Modal(document.getElementById('joinChallengeModal'));
    modal.show();
}

// Submit to challenge
async function submitToChallenge() {
    const description = document.getElementById('challenge-description').value;
    
    if (!description.trim()) {
        alert('Please add a description for your submission');
        return;
    }
    
    const lastTransform = localStorage.getItem('lastTransform');
    
    if (!lastTransform) {
        alert('No transformed image found');
        return;
    }
    
    const submitBtn = document.getElementById('submit-challenge-btn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Submitting...';
    
    try {
        // API call to submit to challenge
        const response = await fetch('/api/challenge/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_path: JSON.parse(lastTransform).image_path,
                style: JSON.parse(lastTransform).style,
                description: description,
                challenge_id: weeklyChallenge?.id
            })
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            alert('✅ Successfully submitted to challenge!');
            
            bootstrap.Modal.getInstance(document.getElementById('joinChallengeModal')).hide();
            document.getElementById('challenge-description').value = '';
            
            // Refresh submissions
            loadSubmissions(true);
        } else {
            alert(data.message || 'Failed to submit');
        }
    } catch (error) {
        console.error('Error submitting:', error);
        alert('Failed to submit. Please try again.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="bi bi-cloud-upload me-2"></i>Submit to Challenge';
    }
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
    try {
        const response = await fetch(`/api/submission/${id}/like`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.code === 200) {
            // Update UI
            const likeBtn = event?.target?.closest('button');
            if (likeBtn) {
                likeBtn.innerHTML = '<i class="bi bi-heart-fill text-danger"></i> Liked';
            }
        }
    } catch (error) {
        console.error('Error liking submission:', error);
    }
}

// Share submission
function shareSubmission(id) {
    const link = `${window.location.origin}/submission/${id}`;
    navigator.clipboard?.writeText(link).then(() => {
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
    
    const submitBtn = document.getElementById('submit-challenge-btn');
    if (submitBtn) {
        submitBtn.addEventListener('click', submitToChallenge);
    }
    
    const joinModal = document.getElementById('joinChallengeModal');
    if (joinModal) {
        joinModal.addEventListener('hidden.bs.modal', () => {
            document.getElementById('challenge-description').value = '';
        });
    }
});