// Add GitHub stars badge after the navbar brand
document.addEventListener('DOMContentLoaded', function() {
    const navbarBrand = document.querySelector('.navbar-brand.logo');
    if (navbarBrand) {
        // Create badge container
        const badge = document.createElement('div');
        badge.className = 'navbar-brand-badge';
        badge.innerHTML = '<a href="https://github.com/brentyi/jaxls"><img src="https://img.shields.io/github/stars/brentyi/jaxls?style=social" alt="GitHub stars"></a>';

        // Insert after navbar brand
        navbarBrand.parentNode.insertBefore(badge, navbarBrand.nextSibling);
    }
});
