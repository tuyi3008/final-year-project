// gallery.js - Gallery with Filter Functionality

// Sample gallery data
const galleryData = [
    {
        id: 1,
        title: "Urban Sketch",
        author: "User123",
        style: "sketch",
        image: "https://via.placeholder.com/400x300/6366f1/ffffff?text=Sketch+Example"
    },
    {
        id: 2,
        title: "Anime Art",
        author: "AnimeFan",
        style: "anime", 
        image: "https://via.placeholder.com/400x300/f59e0b/ffffff?text=Anime+Example"
    },
    {
        id: 3,
        title: "Ink Painting",
        author: "InkMaster",
        style: "ink",
        image: "https://via.placeholder.com/400x300/10b981/ffffff?text=Ink+Example"
    },
    {
        id: 4,
        title: "City Sketch",
        author: "SketchArtist",
        style: "sketch",
        image: "https://via.placeholder.com/400x300/6366f1/ffffff?text=Sketch+2"
    },
    {
        id: 5,
        title: "Fantasy Anime",
        author: "FantasyFan",
        style: "anime",
        image: "https://via.placeholder.com/400x300/f59e0b/ffffff?text=Anime+2"
    },
    {
        id: 6,
        title: "Mountain Ink",
        author: "NatureArtist",
        style: "ink",
        image: "https://via.placeholder.com/400x300/10b981/ffffff?text=Ink+2"
    }
];

let currentFilter = 'all';

// Function to filter gallery items
function filterGallery(filter) {
    currentFilter = filter;
    const filteredData = galleryData.filter(item => {
        if (filter === 'all') return true;
        return item.style === filter;
    });
    renderGallery(filteredData);
}

// Function to render gallery
function renderGallery(data = galleryData) {
    const galleryGrid = document.getElementById('gallery-grid');
    
    if (!galleryGrid) return;
    
    galleryGrid.innerHTML = '';
    
    data.forEach(item => {
        const col = document.createElement('div');
        col.className = 'col-md-4';
        
        col.innerHTML = `
            <div class="gallery-item">
                <img src="${item.image}" alt="${item.title}">
                <div class="gallery-info">
                    <h5>${item.title}</h5>
                    <p>By ${item.author} • ${item.style}</p>
                </div>
            </div>
        `;
        
        galleryGrid.appendChild(col);
    });
}

// Setup filter button events
function setupFilterButtons() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));
            
            // Add active class to clicked button
            button.classList.add('active');
            
            // Filter gallery
            const filter = button.getAttribute('data-filter');
            filterGallery(filter);
        });
    });
}

// Initialize gallery when page loads
document.addEventListener('DOMContentLoaded', () => {
    renderGallery();
    setupFilterButtons();
});