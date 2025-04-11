// State management
let currentMode = 'manual'; // 'manual' or 'verification'
let currentPage = 1;
let isLoading = false;
let hasMore = true;
let searchTimeout = null;

// DOM Elements
const productGrid = document.getElementById('productGrid');
const searchInput = document.getElementById('searchInput');
const manualModeBtn = document.getElementById('manualMode');
const verificationModeBtn = document.getElementById('verificationMode');
const stockModal = document.getElementById('stockModal');
const loading = document.getElementById('loading');
const toast = document.getElementById('toast');
const csvUpload = document.getElementById('csvUpload');
const scrollToTopBtn = document.getElementById('scrollToTop');

const modalProductImage = document.getElementById('modalProductImage');
const modalProductName = document.getElementById('modalProductName');
const modalProductCategory = document.getElementById('modalProductCategory');
const stockAmount = document.getElementById('stockAmount');
const cancelBtn = document.getElementById('cancelBtn');
const saveBtn = document.getElementById('saveBtn');
const addStockBtn = document.getElementById('addStockBtn');
const reduceStockBtn = document.getElementById('reduceStockBtn');
const stockAmountInput = document.querySelector('.stock-amount-input');

// Local state for current product in modal
let currentProduct = null;
let currentOperation = null; // 'add' or 'reduce'

// Event Listeners
document.addEventListener('DOMContentLoaded', init);
searchInput.addEventListener('input', handleSearch);
manualModeBtn.addEventListener('click', () => setMode('manual'));
verificationModeBtn.addEventListener('click', () => setMode('verification'));
csvUpload.addEventListener('change', handleCsvUpload);
window.addEventListener('scroll', handleInfiniteScroll);
window.addEventListener('scroll', toggleScrollToTopButton);
scrollToTopBtn.addEventListener('click', scrollToTop);

// Modal button event listeners
cancelBtn.addEventListener('click', closeModal);
saveBtn.addEventListener('click', saveStock);
addStockBtn.addEventListener('click', () => selectOperation('add'));
reduceStockBtn.addEventListener('click', () => selectOperation('reduce'));

// Direct modal event listener
stockModal.addEventListener('click', function(event) {
    if (event.target === stockModal) {
        closeModal();
    }
});

// Add event listener to close modal when Escape key is pressed
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && !stockModal.hidden) {
        closeModal();
    }
    
    // Add Home key shortcut to scroll to top
    if (event.key === 'Home') {
        scrollToTop();
    }
});

// Initialize the dashboard
async function init() {
    await loadProducts();
    setMode('manual');
}

// Mode switching
function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-toggle button').forEach(btn => btn.classList.remove('active'));
    
    if (mode === 'manual') {
        manualModeBtn.classList.add('active');
        document.querySelectorAll('.verification-checkbox').forEach(checkbox => checkbox.hidden = true);
    } else {
        verificationModeBtn.classList.add('active');
        document.querySelectorAll('.verification-checkbox').forEach(checkbox => checkbox.hidden = false);
    }
}

// Load products from backend
async function loadProducts(search = '') {
    if (isLoading) return;
    
    // If we're loading more products (not the first page) and there are no more, don't load
    if (currentPage > 1 && !hasMore) {
        // Remove any existing loading indicators
        const loadingIndicators = document.querySelectorAll('.loading-more');
        loadingIndicators.forEach(indicator => indicator.remove());
        return;
    }

    isLoading = true;
    loading.hidden = false;

    try {
        const response = await fetch(`/api/products?page=${currentPage}&search=${encodeURIComponent(search)}`);
        const data = await response.json();

        hasMore = data.has_more;
        
        // If this is the first page and no products were found, show the no products message
        if (currentPage === 1 && data.products.length === 0) {
            productGrid.innerHTML = `
                <div class="no-products">
                    <h3>No products found</h3>
                    <p>Try adjusting your search criteria</p>
                </div>
            `;
        } else {
            renderProducts(data.products);
        }
        
        // Only increment the page if we actually got products
        if (data.products.length > 0) {
            currentPage++;
        }
    } catch (error) {
        showToast('Error loading products', 'error');
    } finally {
        isLoading = false;
        loading.hidden = true;
        
        // Remove any loading indicators
        const loadingIndicators = document.querySelectorAll('.loading-more');
        loadingIndicators.forEach(indicator => indicator.remove());
    }
}

// Render product cards
function renderProducts(products) {
    const template = document.getElementById('product-card-template');
    
    // Clear the product grid
    productGrid.innerHTML = '';
    
    // If no products found, show a message
    if (products.length === 0) {
        const noProductsMessage = document.createElement('div');
        noProductsMessage.className = 'no-products';
        noProductsMessage.innerHTML = `
            <h3>No products found</h3>
            <p>Try adjusting your search criteria</p>
        `;
        productGrid.appendChild(noProductsMessage);
        return;
    }

    products.forEach(product => {
        const card = template.content.cloneNode(true);
        const productCard = card.querySelector('.product-card');

        productCard.querySelector('.product-image').src = product.image_url || '/static/images/default.png';
        productCard.querySelector('.product-name').textContent = product.name;
        productCard.querySelector('.product-category').textContent = product.category;

        const stockBadge = productCard.querySelector('.stock-badge');
        stockBadge.textContent = `Stock: ${product.stock}`;
        stockBadge.classList.add(product.stock > 10 ? 'high' : product.stock > 5 ? 'medium' : 'low');

        // Expiry Status
        if (product.expiry_date) {
            const expiryDate = new Date(product.expiry_date);
            const today = new Date();
            const daysUntilExpiry = Math.ceil((expiryDate - today) / (1000 * 60 * 60 * 24));
            const expiryStatus = productCard.querySelector('.expiry-status');

            if (daysUntilExpiry < 0) {
                expiryStatus.textContent = 'EXPIRED';
                expiryStatus.style.color = 'var(--danger-color)';
            } else if (daysUntilExpiry < 30) {
                expiryStatus.textContent = `Expires in ${daysUntilExpiry} days`;
                expiryStatus.style.color = 'var(--warning-color)';
            }
        }

        // Manual Mode Click
        productCard.addEventListener('click', () => {
            if (currentMode === 'manual') openStockModal(product);
        });

        productGrid.appendChild(card);
    });
}

// Debug function to check modal state
function debugModal() {
    console.log('Modal hidden:', stockModal.hidden);
    console.log('Modal display style:', window.getComputedStyle(stockModal).display);
    console.log('Modal has hidden class:', stockModal.classList.contains('hidden'));
    console.log('Current product:', currentProduct);
}

// Select operation (add or reduce)
function selectOperation(operation) {
    currentOperation = operation;
    
    // Update button styles
    addStockBtn.classList.remove('active');
    reduceStockBtn.classList.remove('active');
    if (operation === 'add') {
        addStockBtn.classList.add('active');
    } else {
        reduceStockBtn.classList.add('active');
    }
    
    // Show the amount input
    stockAmountInput.hidden = false;
    saveBtn.hidden = false;
    
    // Reset and focus the input
    stockAmount.value = 0;
    stockAmount.focus();
}

// Close Modal
function closeModal() {
    // Try multiple methods to ensure the modal is hidden
    stockModal.hidden = true;
    stockModal.style.display = 'none';
    stockModal.classList.add('hidden');
    
    // Reset the form
    stockAmount.value = 0;
    currentProduct = null;
    currentOperation = null;
    
    // Reset button states
    addStockBtn.classList.remove('active');
    reduceStockBtn.classList.remove('active');
    stockAmountInput.hidden = true;
    saveBtn.hidden = true;
    
    // Debug
    debugModal();
}

// Open Stock Modal
function openStockModal(product) {
    currentProduct = product;
    modalProductImage.src = product.image_url || '/static/images/default.png';
    modalProductName.textContent = product.name;
    modalProductCategory.textContent = product.category;
    
    // Reset the form
    stockAmount.value = 0;
    currentOperation = null;
    addStockBtn.classList.remove('active');
    reduceStockBtn.classList.remove('active');
    stockAmountInput.hidden = true;
    saveBtn.hidden = true;
    
    // Ensure the modal is visible using multiple methods
    stockModal.hidden = false;
    stockModal.style.display = 'flex';
    stockModal.classList.remove('hidden');
    
    // Debug
    debugModal();
}

// Save Stock Update
async function saveStock() {
    if (!currentOperation) {
        showToast("Please select an operation (Add or Reduce)", "error");
        return;
    }

    const amount = parseInt(stockAmount.value);
    if (isNaN(amount) || amount <= 0) {
        showToast("Please enter a valid amount", "error");
        return;
    }

    const delta = currentOperation === 'add' ? amount : -amount;

    try {
        const response = await fetch(`/api/update_stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                product_id: currentProduct.id,
                delta: delta
            })
        });

        const result = await response.json();
        if (result.success) {
            showToast("Stock updated successfully!", "success");
            
            // Close the modal first
            closeModal();
            
            // Reset pagination and reload products
            productGrid.innerHTML = '';
            currentPage = 1;
            hasMore = true;
            
            // Force a complete reload of products
            await loadProducts(searchInput.value.trim());
            
            // Force a second reload after a short delay to ensure the backend has updated
            setTimeout(async () => {
                await loadProducts(searchInput.value.trim());
            }, 500);
        } else {
            showToast(result.message || "Failed to update stock", "error");
        }
    } catch (error) {
        showToast("Error updating stock", "error");
        console.error('Error:', error);
    }
}

// CSV Upload Handler
async function handleCsvUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/import_csv", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        if (result.success) {
            showToast("CSV uploaded successfully!", "success");
            productGrid.innerHTML = '';
            currentPage = 1;
            hasMore = true;
            await loadProducts(searchInput.value.trim());
        } else {
            showToast(result.message || "Upload failed", "error");
        }
    } catch (error) {
        showToast("Upload error", "error");
    }
}

// Infinite Scrolling
function handleInfiniteScroll() {
    // Check if we're near the bottom of the page
    if ((window.innerHeight + window.scrollY) >= document.documentElement.scrollHeight - 500) {
        // Only load more if we're not already loading and there's more to load
        if (!isLoading && hasMore) {
            // Show loading indicator at the bottom
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading-more';
            loadingIndicator.innerHTML = `
                <div class="spinner"></div>
                <p>Loading more products...</p>
            `;
            productGrid.appendChild(loadingIndicator);
            
            // Load more products
            loadProducts(searchInput.value.trim());
        }
    }
}

// Debounced Search
function handleSearch() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        // Reset pagination
        currentPage = 1;
        hasMore = true;
        
        // Clear the product grid and show loading
        productGrid.innerHTML = '';
        loading.hidden = false;
        
        // Load products with the search term
        loadProducts(searchInput.value.trim());
    }, 400);
}

// Toast Notifications
function showToast(message, type = 'success') {
    toast.textContent = message;
    toast.style.backgroundColor = type === 'success' ? 'var(--success-color)' : 'var(--danger-color)';
    toast.hidden = false;
    setTimeout(() => toast.hidden = true, 3000);
}

// Toggle scroll to top button visibility
function toggleScrollToTopButton() {
    if (window.scrollY > 300) {
        scrollToTopBtn.hidden = false;
    } else {
        scrollToTopBtn.hidden = true;
    }
}

// Scroll to top function
function scrollToTop() {
    // Use a more reliable method to scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
    
    // Also try the alternative method as a fallback
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
    
    // Hide the button after scrolling
    scrollToTopBtn.hidden = true;
}
