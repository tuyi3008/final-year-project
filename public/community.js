// community.js - Slightly Improved

// Sample posts
const posts = [
    {
        user: "SketchMaster",
        text: "Just finished a cityscape sketch using our pencil style. The details came out amazing!",
        likes: 42,
        comments: 8,
        time: "2 hours ago"
    },
    {
        user: "AnimeFan", 
        text: "Trying out the anime style for portrait transformation. Loving the vibrant colors!",
        likes: 89,
        comments: 15,
        time: "1 day ago"
    },
    {
        user: "InkArtist",
        text: "Experimenting with ink wash techniques on landscape photos. The results are stunning!",
        likes: 56,
        comments: 12,
        time: "3 days ago"
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
                <div class="post-author">
                    <i class="bi bi-person-circle"></i>
                    ${post.user}
                </div>
                <div class="post-content">${post.text}</div>
                <div class="post-stats">
                    <span><i class="bi bi-heart"></i> ${post.likes}</span>
                    <span><i class="bi bi-chat"></i> ${post.comments}</span>
                    <span>${post.time}</span>
                </div>
            `;
            
            container.appendChild(postDiv);
        });
    }
});