// community.js - Simple Community Script

// Sample posts
const posts = [
    {
        user: "SketchMaster",
        text: "Just created an amazing sketch!",
        likes: 42,
        time: "2h"
    },
    {
        user: "AnimeFan", 
        text: "Anime style transformation tutorial",
        likes: 89,
        time: "1d"
    },
    {
        user: "InkArtist",
        text: "Traditional ink painting techniques",
        likes: 56,
        time: "3d"
    }
];

// Load posts on page load
window.addEventListener('DOMContentLoaded', () => {
    const container = document.querySelector('.community-container');
    
    if (container) {
        posts.forEach(post => {
            const postDiv = document.createElement('div');
            postDiv.className = 'community-post';
            
            postDiv.innerHTML = `
                <div class="post-author">${post.user}</div>
                <div class="post-content">${post.text}</div>
                <div class="post-stats">
                    ${post.likes} likes • ${post.time} ago
                </div>
            `;
            
            container.appendChild(postDiv);
        });
    }
});