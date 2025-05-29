// Global variables
let products = [];
let categories = new Set();

// Initialize the page
function initializePage() {
    loadProducts();
    setupEventListeners();
}

// Load all products
function loadProducts(searchQuery = '') {
    const category = document.getElementById('categoryFilter').value;
    const stockStatus = document.getElementById('stockFilter').value;
    
    // Build query parameters
    const params = new URLSearchParams();
    if (searchQuery) params.append('q', searchQuery);
    if (category) params.append('category', category);
    if (stockStatus) params.append('stock_status', stockStatus);

    // Fetch products from API
    fetch(`/api/products/search?${params.toString()}`)
        .then(response => response.json())
        .then(products => {
            const productGrid = document.getElementById('productGrid');
            productGrid.innerHTML = '';

            products.forEach(product => {
                const card = createProductCard(product);
                productGrid.appendChild(card);
            });
        })
        .catch(error => {
            console.error('Error loading products:', error);
            alert('Error loading products. Please try again.');
        });
}

// Update product dropdowns in the sale form
function updateProductDropdowns() {
    const productSelects = document.querySelectorAll('select[name="products[]"]');
    productSelects.forEach(select => {
        select.innerHTML = '<option value="">Select Product</option>';
        products.forEach(product => {
            if (product.current_stock > 0) {
                const option = document.createElement('option');
                option.value = product.id;
                option.textContent = `${product.name} (${product.brand}) - ₹${product.unit_price}`;
                option.dataset.price = product.unit_price;
                option.dataset.gst = product.gst_percentage;
                select.appendChild(option);
            }
        });
    });
}

// Update category filter dropdown
function updateCategoryFilter() {
    const categoryFilter = document.getElementById('categoryFilter');
    categories.clear();
    products.forEach(product => categories.add(product.category));
    
    categoryFilter.innerHTML = '<option value="">All Categories</option>';
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        categoryFilter.appendChild(option);
    });
}

// Display products in the product grid
function displayProducts(filteredProducts = products) {
    const productGrid = document.getElementById('productGrid');
    productGrid.innerHTML = '';
    
    filteredProducts.forEach(product => {
        const card = document.createElement('div');
        card.className = 'col-md-4 mb-4';
        card.innerHTML = `
            <div class="product-card">
                <h5>${product.name}</h5>
                <p class="mb-1">Brand: ${product.brand}</p>
                <p class="mb-1">Category: ${product.category}</p>
                <p class="mb-1">Price: ₹${product.unit_price}</p>
                <p class="mb-1">GST: ${product.gst_percentage}%</p>
                <p class="mb-1">Stock: ${product.current_stock}</p>
                <span class="stock-status stock-${product.stock_status}">
                    ${product.stock_status.toUpperCase()}
                </span>
            </div>
        `;
        productGrid.appendChild(card);
    });
}

// Filter products based on search and filters
function filterProducts() {
    const searchQuery = document.getElementById('productSearch').value.toLowerCase();
    const categoryFilter = document.getElementById('categoryFilter').value;
    const stockFilter = document.getElementById('stockFilter').value;
    
    const filteredProducts = products.filter(product => {
        const matchesSearch = product.name.toLowerCase().includes(searchQuery) ||
                            product.brand.toLowerCase().includes(searchQuery);
        const matchesCategory = !categoryFilter || product.category === categoryFilter;
        const matchesStock = !stockFilter || product.stock_status === stockFilter;
        
        return matchesSearch && matchesCategory && matchesStock;
    });
    
    displayProducts(filteredProducts);
}

// Calculate order summary
function calculateOrderSummary() {
    const orderSummary = document.getElementById('orderSummary');
    const productItems = document.querySelectorAll('.product-item');
    let subtotal = 0;
    let totalGST = 0;
    let total = 0;
    
    productItems.forEach(item => {
        const productSelect = item.querySelector('select[name="products[]"]');
        const quantityInput = item.querySelector('input[name="quantities[]"]');
        
        if (productSelect.value && quantityInput.value) {
            const product = products.find(p => p.id === parseInt(productSelect.value));
            const quantity = parseInt(quantityInput.value);
            
            if (product) {
                const itemSubtotal = product.unit_price * quantity;
                const itemGST = itemSubtotal * (product.gst_percentage / 100);
                
                subtotal += itemSubtotal;
                totalGST += itemGST;
            }
        }
    });
    
    total = subtotal + totalGST;
    
    orderSummary.innerHTML = `
        <div class="row">
            <div class="col-6">Subtotal:</div>
            <div class="col-6 text-end">₹${subtotal.toFixed(2)}</div>
        </div>
        <div class="row">
            <div class="col-6">GST:</div>
            <div class="col-6 text-end">₹${totalGST.toFixed(2)}</div>
        </div>
        <div class="row fw-bold">
            <div class="col-6">Total:</div>
            <div class="col-6 text-end">₹${total.toFixed(2)}</div>
        </div>
    `;
}

// Handle sale form submission
function handleSaleSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const items = [];
    
    document.querySelectorAll('.product-item').forEach(item => {
        const productId = item.querySelector('select[name="products[]"]').value;
        const quantity = item.querySelector('input[name="quantities[]"]').value;
        
        if (productId && quantity) {
            items.push({
                product_id: parseInt(productId),
                quantity: parseInt(quantity)
            });
        }
    });
    
    const saleData = {
        customer_name: formData.get('customer_name'),
        phone: formData.get('phone'),
        payment_method: formData.get('payment_method'),
        items: items
    };
    
    fetch('/api/sales', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(saleData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            alert('Sale completed successfully!');
            event.target.reset();
            loadProducts();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the sale.');
    });
}

// Handle restock form submission
function handleRestockSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const productData = {
        product_name: formData.get('product_name'),
        brand_name: formData.get('brand_name'),
        category: formData.get('category'),
        subcategory: formData.get('subcategory'),
        unit_price: formData.get('unit_price'),
        cost_price: formData.get('cost_price'),
        gst_percentage: formData.get('gst_percentage'),
        current_stock: formData.get('current_stock'),
        reorder_level: formData.get('reorder_level'),
        reorder_quantity: formData.get('reorder_quantity'),
        supplier_name: formData.get('supplier_name'),
        lead_time: formData.get('lead_time')
    };
    
    fetch('/api/products', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(productData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            alert('Product saved successfully!');
            event.target.reset();
            loadProducts();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while saving the product.');
    });
}

// Setup event listeners
function setupEventListeners() {
    // Sale form
    document.getElementById('saleForm').addEventListener('submit', handleSaleSubmit);
    document.getElementById('addProduct').addEventListener('click', () => {
        const productList = document.getElementById('productList');
        const newItem = productList.children[0].cloneNode(true);
        newItem.querySelector('select').value = '';
        newItem.querySelector('input').value = '';
        productList.appendChild(newItem);
        updateProductDropdowns();
    });
    
    // Restock form
    document.getElementById('restockForm').addEventListener('submit', handleRestockSubmit);
    
    // Product search and filters
    document.getElementById('productSearch').addEventListener('input', filterProducts);
    document.getElementById('categoryFilter').addEventListener('change', filterProducts);
    document.getElementById('stockFilter').addEventListener('change', filterProducts);
    
    // Calculate order summary when products or quantities change
    document.getElementById('productList').addEventListener('change', calculateOrderSummary);

    // View Report button
    document.getElementById('viewReport').addEventListener('click', function() {
        window.location.href = '/analytics';
    });
}

// Initialize the page when the DOM is loaded
document.addEventListener('DOMContentLoaded', initializePage);

function createProductCard(product) {
    const col = document.createElement('div');
    col.className = 'col-md-4 mb-4';

    const stockStatusClass = {
        'low': 'stock-low',
        'ok': 'stock-ok',
        'out': 'stock-out'
    }[product.stock_status] || 'stock-out';

    col.innerHTML = `
        <div class="product-card">
            <h5 class="card-title">${product.name}</h5>
            <p class="card-text">
                <strong>Brand:</strong> ${product.brand}<br>
                <strong>Category:</strong> ${product.category}<br>
                <strong>Price:</strong> ₹${product.unit_price.toFixed(2)}<br>
                <strong>Stock:</strong> ${product.current_stock} units<br>
                <span class="stock-status ${stockStatusClass}">
                    ${product.stock_status.toUpperCase()}
                </span>
            </p>
            <div class="d-flex justify-content-between">
                <button class="btn btn-sm btn-primary" onclick="editProduct(${product.id})">
                    <i class="fas fa-edit"></i> Edit
                </button>
                <button class="btn btn-sm btn-success" onclick="restockProduct(${product.id})">
                    <i class="fas fa-boxes"></i> Restock
                </button>
            </div>
        </div>
    `;

    return col;
}

function editProduct(productId) {
    // Switch to restock tab and populate form
    document.getElementById('restock-tab').click();
    // TODO: Implement form population
}

function restockProduct(productId) {
    // Switch to restock tab and populate form
    document.getElementById('restock-tab').click();
    // TODO: Implement form population
} 