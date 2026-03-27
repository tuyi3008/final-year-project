// static/js/auth.js

class AuthManager {
    constructor() {
        console.log('AuthManager constructor');

        this.profileCache = null;
        this.profileCacheTime = 0;
        this.cacheTTL = 60000;
        this.isRefreshingXP = false;  // prvent multiple simultaneous XP refreshes
        
        // make sure init runs after DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }
    
    init() {
        console.log('AuthManager init');
        
        // Bind login buttons (in case they exist at init)
        this.bindLoginButtons();
        
        // Global click listener for dynamically added login buttons
        document.addEventListener('click', (e) => {
            const loginButton = e.target.closest('.btn-login, .auth-button, [data-action="login"]');
            if (loginButton) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Login button clicked (global)');
                this.showLoginModal();
            }
        });
        
        // Check login status on page load
        this.checkLoginStatus();
        
        // Update UI based on login status
        this.updateUI();
        
        // Load user avatar
        this.loadUserAvatar();

        this.initXPEvents();
        
        // Listen for avatar update events
        window.addEventListener('avatarUpdated', (event) => {
            const { avatarPath } = event.detail;
            this.updateAvatarDisplay(avatarPath);
        });
    }

    initXPEvents() {
        window.addEventListener('xpUpdated', (event) => {
            console.log('XP updated event received:', event.detail);

            // Prevent multiple simultaneous XP refreshes
            if (this.isRefreshingXP) {
                console.log('⏭️ Skipping XP refresh (already refreshing)');
                return;
            }

            if (window.refreshXP) {
                window.refreshXP();
            }

            document.dispatchEvent(new CustomEvent('userXPChanged', { 
                detail: event.detail 
            }));
        });
        
        console.log('XP events initialized');
    }

    showXPMessage(message, type = 'info') {
        this.showMessage(message, type);
    }

    async _getProfile() {
        if (!this.isLoggedIn()) return null;
        
        const now = Date.now();
        if (this.profileCache && (now - this.profileCacheTime) < this.cacheTTL) {
            console.log('📦 Using cached profile');
            return this.profileCache;
        }
        
        try {
            const response = await fetch('http://localhost:8000/api/user/profile', {
                headers: this.getAuthHeaders()
            });
            
            if (response.status === 401) {
                console.log('Token expired, logging out...');
                this.logout();
                return null;
            }
            
            const data = await response.json();
            if (data.code === 200) {
                this.profileCache = data;
                this.profileCacheTime = now;
                console.log('✅ Profile cached');
                return data;
            }
            return null;
        } catch (error) {
            console.error('Error fetching profile:', error);
            return null;
        }
    }

    async getUserXP() {
        if (!this.isLoggedIn()) return 0;
        
        const data = await this._getProfile();
        return data?.total_xp || 0;
    }

    async refreshAllXPDisplays() {
        if (!this.isLoggedIn()) return;

        if (this.isRefreshingXP) {
            console.log('⏭️ Skipping XP refresh (already in progress)');
            return;
        }
        
        this.isRefreshingXP = true;
        
        try {
            this.profileCache = null;
            this.profileCacheTime = 0;
            
            const xp = await this.getUserXP();

            window.dispatchEvent(new CustomEvent('xpUpdated', { 
                detail: { amount: 0, source: 'refresh', total: xp }
            }));
            
            return xp;
        } finally {
            this.isRefreshingXP = false;
        }
    }
    
    bindLoginButtons() {
        const loginButtons = document.querySelectorAll('.btn-login, .auth-button, [data-action="login"]');
        console.log('Found login buttons:', loginButtons.length);
        
        loginButtons.forEach(btn => {
            // remove old listeners to prevent duplicates
            btn.removeEventListener('click', this.boundHandler);
            // add new listener
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Login button clicked (direct)');
                this.showLoginModal();
            });
        });
    }
    
    async showLoginModal() {
        console.log('showLoginModal called');
        
        // Check if modal already exists
        let modal = document.getElementById('authModal');
        
        if (!modal) {
            // Load modal template
            try {
                const response = await fetch('/static/templates/auth_modal.html');
            
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const html = await response.text();
                document.body.insertAdjacentHTML('beforeend', html);
                this.attachEvents();
                modal = document.getElementById('authModal');
                
                if (!modal) {
                    throw new Error('Modal element not found after insertion');
                }
                
            } catch (error) {
                console.error('Failed to load auth modal:', error);
                alert('Failed to load login form. Please refresh the page.');
                return;
            }
        }
        
        // Show modal
        try {
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
        } catch (error) {
            console.error('Failed to show modal:', error);
            alert('Bootstrap modal error. Make sure Bootstrap is loaded.');
        }
    }
    
    attachEvents() {
        console.log('Attaching form events');
        
        // Login form submission
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            // Remove any existing listeners to prevent duplicates
            loginForm.removeEventListener('submit', this.boundLoginHandler);
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
            console.log('Login form event attached');
        } else {
            console.warn('Login form not found');
        }
        
        // Register form submission
        const registerForm = document.getElementById('registerForm');
        if (registerForm) {
            registerForm.removeEventListener('submit', this.boundRegisterHandler);
            registerForm.addEventListener('submit', (e) => this.handleRegister(e));
            console.log('Register form event attached');
        } else {
            console.warn('Register form not found');
        }
        
        // Social login buttons
        const googleBtn = document.getElementById('googleLoginBtn');
        if (googleBtn) {
            googleBtn.addEventListener('click', () => this.socialLogin('google'));
        }
        
        const githubBtn = document.getElementById('githubLoginBtn');
        if (githubBtn) {
            githubBtn.addEventListener('click', () => this.socialLogin('github'));
        }
        
        // Password confirmation validation
        const password = document.getElementById('registerPassword');
        const confirmPassword = document.getElementById('confirmPassword');
        if (password && confirmPassword) {
            confirmPassword.addEventListener('input', () => this.validatePasswordMatch());
        }
    }
    
    async handleLogin(e) {
        e.preventDefault();
        console.log('Login form submitted');
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        if (!email || !password) {
            this.showMessage('Please enter email and password', 'error');
            return;
        }
        
        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `username=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`
            });
            
            const data = await response.json();
            console.log('Login response:', data);
            
            if (response.ok && data.access_token) {
                // Save token
                localStorage.setItem('token', data.access_token);
                localStorage.setItem('token_type', data.token_type);
                localStorage.setItem('userEmail', email);
                
                this.showMessage('Login successful!', 'success');
                
                document.dispatchEvent(new CustomEvent('userLoggedIn', { 
                    detail: { email: email } 
                }));
                
                // Close modal after 1 second
                setTimeout(() => {
                    const modalElement = document.getElementById('authModal');
                    if (modalElement) {
                        const modal = bootstrap.Modal.getInstance(modalElement);
                        if (modal) modal.hide();
                    }
                    this.profileCache = null;
                    this.profileCacheTime = 0;
                    this.updateUI();
                    // Load avatar after successful login
                    this.loadUserAvatar();

                    this.refreshAllXPDisplays();
                    
                }, 1000);
                
            } else {
                this.showMessage(data.detail || 'Login failed', 'error');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        }
    }
    
    async handleRegister(e) {
        e.preventDefault();
        console.log('Register form submitted');
        
        const firstName = document.getElementById('firstName')?.value || '';
        const lastName = document.getElementById('lastName')?.value || '';
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const termsAgree = document.getElementById('termsAgree').checked;
        
        // Validation
        if (password !== confirmPassword) {
            this.showMessage('Passwords do not match', 'error');
            return;
        }
        
        if (!termsAgree) {
            this.showMessage('Please agree to Terms of Service', 'error');
            return;
        }
        
        if (password.length < 8) {
            this.showMessage('Password must be at least 8 characters', 'error');
            return;
        }
        
        // Create username from first and last name
        const username = firstName || lastName ? 
            `${firstName.toLowerCase()}.${lastName.toLowerCase()}` : 
            email.split('@')[0];
        
        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password,
                    email: email
                })
            });
            
            const data = await response.json();
            console.log('Register response:', data);
            
            if (response.ok) {
                this.showMessage('Registration successful! Please login.', 'success');
                
                // Clear form and switch to login tab
                document.getElementById('registerForm').reset();
                
                // Switch to login tab
                const loginTab = document.getElementById('login-tab');
                if (loginTab) {
                    const tab = new bootstrap.Tab(loginTab);
                    tab.show();
                }
                
                // Pre-fill login email
                document.getElementById('loginEmail').value = email;
                
            } else {
                this.showMessage(data.detail || 'Registration failed', 'error');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        }
    }
    
    validatePasswordMatch() {
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const confirmInput = document.getElementById('confirmPassword');
        
        if (confirmPassword && password !== confirmPassword) {
            confirmInput.setCustomValidity('Passwords do not match');
            confirmInput.classList.add('is-invalid');
        } else {
            confirmInput.setCustomValidity('');
            confirmInput.classList.remove('is-invalid');
        }
    }
    
    socialLogin(provider) {
        this.showMessage(`${provider} login coming soon!`, 'info');
    }
    
    showMessage(message, type = 'info') {
        const container = document.getElementById('authMessageContainer');
        if (!container) {
            console.warn('Message container not found');
            alert(message);
            return;
        }
        
        const messageId = 'msg-' + Date.now();
        const messageHtml = `
            <div id="${messageId}" class="auth-alert ${type}">
                <span>${message}</span>
                <button class="close-btn" onclick="this.parentElement.remove()">&times;</button>
            </div>
        `;
        
        container.style.display = 'block';
        container.insertAdjacentHTML('beforeend', messageHtml);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            const msgElement = document.getElementById(messageId);
            if (msgElement) msgElement.remove();
            if (container.children.length === 0) {
                container.style.display = 'none';
            }
        }, 3000);
    }
    
    // ====================== Avatar Functions ======================
    
    // Load user avatar from server
    async loadUserAvatar() {
        if (!this.isLoggedIn()) return;
        
        console.log('Loading user avatar...');
        
        // 🆕 使用缓存方法
        const data = await this._getProfile();
        
        if (data && data.code === 200 && data.avatar_path) {
            this.updateAvatarDisplay(data.avatar_path);
        } else {
            // Set default avatar if no avatar found
            this.setDefaultAvatar();
        }
    }
    
    // Update avatar display across all pages
    updateAvatarDisplay(avatarPath) {
        console.log('Updating avatar display:', avatarPath);
        
        // Update navbar user menu avatar
        const navbarAvatar = document.querySelector('.user-icon img');
        if (navbarAvatar) {
            navbarAvatar.src = `/${avatarPath}`;
            navbarAvatar.onerror = () => this.setDefaultAvatar();
        }
        
        // Update profile page avatar if it exists
        const profileAvatar = document.querySelector('.profile-avatar img');
        if (profileAvatar) {
            profileAvatar.src = `/${avatarPath}`;
            profileAvatar.onerror = () => this.setDefaultAvatar();
        }
        
        // Update any other avatar elements with class 'user-avatar'
        const userAvatars = document.querySelectorAll('.user-avatar');
        userAvatars.forEach(img => {
            img.src = `/${avatarPath}`;
            img.onerror = () => this.setDefaultAvatar();
        });
        
        // Trigger event for other components
        window.dispatchEvent(new CustomEvent('avatarUpdated', { 
            detail: { avatarPath: avatarPath } 
        }));
    }
    
    // Set default avatar (Bootstrap person icon as SVG)
    setDefaultAvatar() {
        const defaultSvg = 'data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2240%22%20height%3D%2240%22%20viewBox%3D%220%200%2016%2016%22%3E%3Cpath%20fill%3D%22%236c757d%22%20d%3D%22M8%208a3%203%200%201%200%200-6%203%203%200%200%200%200%206zm2-3a2%202%200%201%201-4%200%202%202%200%200%201%204%200zm4%208c0%201-1%201-1%201H3s-1%200-1-1%201-4%206-4%206%203%206%204zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516%2010.68%2010.289%2010%208%2010c-2.29%200-3.516.68-4.168%201.332-.678.678-.83%201.418-.832%201.664h10z%22%2F%3E%3C%2Fsvg%3E';
        
        const navbarAvatar = document.querySelector('.user-icon img');
        if (navbarAvatar && navbarAvatar.src !== defaultSvg) {
            navbarAvatar.src = defaultSvg;
        }
        
        // Also update profile avatar if exists
        const profileAvatar = document.querySelector('.profile-avatar img');
        if (profileAvatar && profileAvatar.src !== defaultSvg) {
            profileAvatar.src = defaultSvg;
        }
    }
    
    // Get avatar HTML for dynamic insertion
    getAvatarHTML(size = 40) {
        const token = localStorage.getItem('token');
        if (!token) return '';
        
        return `<img src="/static/images/default-avatar.png" 
                     alt="User Avatar" 
                     class="rounded-circle user-avatar"
                     style="width: ${size}px; height: ${size}px; object-fit: cover;"
                     onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22${size}%22%20height%3D%22${size}%22%20viewBox%3D%220%200%2016%2016%22%3E%3Cpath%20fill%3D%22%236c757d%22%20d%3D%22M8%208a3%203%200%201%200%200-6%203%203%200%200%200%200%206zm2-3a2%202%200%201%201-4%200%202%202%200%200%201%204%200zm4%208c0%201-1%201-1%201H3s-1%200-1-1%201-4%206-4%206%203%206%204zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516%2010.68%2010.289%2010%208%2010c-2.29%200-3.516.68-4.168%201.332-.678.678-.83%201.418-.832%201.664h10z%22%2F%3E%3C%2Fsvg%3E';">`;
    }
    
    // ====================== Existing Functions ======================
    
    isLoggedIn() {
        return !!localStorage.getItem('token');
    }
    
    getAuthHeaders() {
        const token = localStorage.getItem('token');
        const tokenType = localStorage.getItem('token_type') || 'Bearer';
        
        return token ? {
            'Authorization': `${tokenType} ${token}`
        } : {};
    }
    
    updateUI() {
        const isLoggedIn = this.isLoggedIn();
        console.log('Updating UI, logged in:', isLoggedIn);
        
        // Update all login buttons
        document.querySelectorAll('.btn-login, .auth-button').forEach(el => {
            el.style.display = isLoggedIn ? 'none' : 'inline-block';
        });
        
        // Update user menu if exists
        document.querySelectorAll('.user-menu, .profile-link').forEach(el => {
            el.style.display = isLoggedIn ? 'inline-block' : 'none';
        });
        
        // Load avatar if logged in, set default if logged out
        if (isLoggedIn) {
            this.loadUserAvatar();
            this.refreshAllXPDisplays();
        } else {
            this.setDefaultAvatar();
            this.clearXPDisplay();
        }

        const logoutBtn = document.querySelector('.btn-logout');
        if (logoutBtn) {
            // remove old listener to prevent duplicates
            logoutBtn.removeEventListener('click', this.logout);
            // add new listener
            logoutBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.logout();
            });
            console.log('Logout button event attached');
        }
    }

    clearXPDisplay() {
        const xpElements = document.querySelectorAll('.xp-display, .level-display, #userLevel, #totalPoints');
        xpElements.forEach(el => {
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                el.value = '';
            } else {
                el.textContent = '0';
            }
        });

        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            bar.style.width = '0%';
        });
    }
    
    logout() {
        localStorage.removeItem('token');
        localStorage.removeItem('token_type');
        localStorage.removeItem('userEmail');

        this.profileCache = null;
        this.profileCacheTime = 0;

        document.dispatchEvent(new CustomEvent('userLoggedOut'));
        
        this.updateUI();
        this.setDefaultAvatar(); // Reset to default avatar
        this.showMessage('Logged out successfully', 'success');
        
        // Redirect to home if on protected page
        if (window.location.pathname.includes('profile') || 
            window.location.pathname.includes('gallery') ||
            window.location.pathname.includes('community')) {
            window.location.href = '/';
        }
    }
    
    checkLoginStatus() {
        // Optional: Verify token with backend
        const token = localStorage.getItem('token');
        if (token) {
            console.log('User is logged in');
            // Could add token validation with backend here
        }
    }
}

// Initialize auth manager
console.log('Creating AuthManager instance');
const auth = new AuthManager();

// Make auth available globally
console.log('Before window.auth assignment, auth exists:', !!auth);
window.auth = auth;
console.log('After assignment, window.auth exists:', !!window.auth);

window.refreshXP = auth.refreshAllXPDisplays.bind(auth);
window.getUserXP = auth.getUserXP.bind(auth);