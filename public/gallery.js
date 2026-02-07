// gallery.js - Basic Gallery Functionality

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
    }
];

// Function to load gallery
function loadGallery() {
    const galleryGrid = document.getElementById('gallery-grid');
    
    if (!galleryGrid) return;
    
    galleryData.forEach(item => {
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

// Load gallery when page loads
document.addEventListener('DOMContentLoaded', loadGallery);