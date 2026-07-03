// --- Configuration & Initialization ---
let datasetLoaded = true; // API is always ready

document.addEventListener('DOMContentLoaded', () => {
    initPipelineObserver();
    fetchCategories();
});

// --- DOM Elements ---
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const categorySelect = document.getElementById('categorySelect');
const sampleChips = document.querySelectorAll('.sample-chip');
const selectedProductPanel = document.getElementById('selectedProductPanel');
const selectedProductContent = document.getElementById('selectedProductContent');
const loadingState = document.getElementById('loadingState');
const dashboardContent = document.getElementById('dashboardContent');

// Modal Elements
const openAiModalBtn = document.getElementById('openAiModalBtn');
const aiModal = document.getElementById('aiModal');
const closeAiModalBtn = document.getElementById('closeAiModalBtn');
const aiModalBackdrop = document.getElementById('aiModalBackdrop');

// Tiers
const highTierResults = document.getElementById('highTierResults');
const mediumTierResults = document.getElementById('mediumTierResults');
const lowTierResults = document.getElementById('lowTierResults');

// Metrics
const scoreChart = document.getElementById('scoreChart');
const precisionRing = document.getElementById('precisionRing');
const precisionValue = document.getElementById('precisionValue');

// AI Elements
const aiGoalInput = document.getElementById('aiGoalInput');
const aiGoalBtn = document.getElementById('aiGoalBtn');
const aiDashboardContent = document.getElementById('aiDashboardContent');
const aiGoalTitle = document.getElementById('aiGoalTitle');
const aiResultsGrid = document.getElementById('aiResultsGrid');


// --- Event Listeners ---
sampleChips.forEach(chip => {
    chip.addEventListener('click', (e) => {
        const query = e.target.getAttribute('data-product');
        searchInput.value = query;
        triggerSearch(query, null, categorySelect.value);
    });
});

searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        triggerSearch(searchInput.value.trim(), null, categorySelect.value);
    }
});

if (searchBtn) {
    searchBtn.addEventListener('click', () => {
        triggerSearch(searchInput.value.trim(), null, categorySelect.value);
    });
}

categorySelect.addEventListener('change', () => {
    // If a category is selected, we can immediately fetch results for it
    triggerSearch(searchInput.value.trim(), null, categorySelect.value);
});

aiGoalBtn.addEventListener('click', () => {
    const goal = aiGoalInput.value.trim();
    if (!goal) return;

    // Close the modal
    if (aiModal) {
        aiModal.classList.add('hidden');
    }

    triggerAIProject(goal, '');
});

aiGoalInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && aiGoalInput.value.trim() !== '') {
        const goal = aiGoalInput.value.trim();
        if (aiModal) aiModal.classList.add('hidden');
        triggerAIProject(goal, '');
    }
});

// Modal Logic
if (openAiModalBtn && aiModal && closeAiModalBtn && aiModalBackdrop) {
    const openModal = (e) => {
        e.preventDefault();
        aiModal.classList.remove('hidden');
        setTimeout(() => aiGoalInput.focus(), 100);
    };
    
    const closeModal = (e) => {
        e.preventDefault();
        aiModal.classList.add('hidden');
    };

    openAiModalBtn.addEventListener('click', openModal);
    closeAiModalBtn.addEventListener('click', closeModal);
    aiModalBackdrop.addEventListener('click', closeModal);
}

// --- Core UI Logic ---
function fetchCategories() {
    const hostname = window.location.hostname || '127.0.0.1';
    fetch(`http://${hostname}:5000/api/categories`)
        .then(res => res.json())
        .then(data => {
            if (Array.isArray(data)) {
                data.forEach(cat => {
                    const option = document.createElement('option');
                    option.value = cat.id;
                    option.textContent = cat.category_name;
                    categorySelect.appendChild(option);
                });
            }
        })
        .catch(err => console.error("Failed to load categories:", err));
}

function triggerSearch(query, asin = null, categoryId = null) {
    // Check if we have anything to search with
    if (!query && !asin && !categoryId) return;

    // 1. Hide current results if any, show loading
    dashboardContent.classList.add('hidden');
    if (aiDashboardContent) aiDashboardContent.classList.add('hidden');
    selectedProductPanel.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingState.querySelector('p').textContent = "Computing similarity scores in high-dimensional space...";

    const hostname = window.location.hostname || '127.0.0.1';
    let apiUrl = `http://${hostname}:5000/api/recommend?`;
    
    let params = new URLSearchParams();
    if (query) params.append('q', query);
    if (asin) params.append('asin', asin);
    if (categoryId) params.append('category_id', categoryId);

    apiUrl += params.toString();

    // 2. Compute Recommendations (via real API)
    fetch(apiUrl)
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || "API Error"); });
            }
            return response.json();
        })
        .then(simData => {
            loadingState.classList.add('hidden');
            renderSelectedProduct(simData.target);
            
            // Show dashboard container
            dashboardContent.classList.remove('hidden');
            
            // 3. Render and Animate Tiers with staggered delay
            renderTier(highTierResults, simData.recommendations.high, 'high');
            
            setTimeout(() => {
                renderTier(mediumTierResults, simData.recommendations.medium, 'medium');
            }, 300); // 0.3s delay
            
            setTimeout(() => {
                renderTier(lowTierResults, simData.recommendations.low, 'low');
            }, 600); // 0.6s delay
            
            // 4. Render Chart
            setTimeout(() => {
                renderChart(simData.recommendations);
            }, 800);
            
            // 5. Animate Evaluation Ring
            setTimeout(() => {
                animatePrecisionRing(simData.precision);
            }, 1000);
            
        })
        .catch(error => {
            console.error(error);
            alert("Search Failed: " + error.message);
            loadingState.classList.add('hidden');
        });
}

function renderSelectedProduct(product) {
    const imageUrl = product.imgUrl || 'https://via.placeholder.com/200?text=No+Image';
    selectedProductContent.innerHTML = `
        <div class="target-details flex flex-col gap-4">
            <div class="w-full h-64 bg-surface-container-low rounded-lg overflow-hidden border border-outline-variant relative">
                <img src="${imageUrl}" class="w-full h-full object-contain" onerror="this.src='https://via.placeholder.com/200?text=No+Image'">
            </div>
            <div class="target-info flex-1 min-w-0">
                <h3 class="font-h3 text-h3 text-on-surface mb-xs" title="${product.name}">${product.name}</h3>
                <p class="font-body-sm text-body-sm text-on-surface-variant mb-md truncate" title="${product.category}">${product.category}</p>
                <div class="flex justify-between items-center py-sm border-t border-outline-variant">
                    <div>
                        <p class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider mb-1">Price</p>
                        <p class="font-h4 text-h4 text-on-surface">${product.price}</p>
                    </div>
                    <div class="text-right">
                        <p class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider mb-1">Rating</p>
                        <div class="flex items-center justify-end">
                            <span class="material-symbols-outlined text-tertiary-container mr-1 text-[18px]">star</span>
                            <span class="font-h4 text-h4 text-on-surface">${product.rating}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    selectedProductPanel.classList.remove('hidden');
}

function renderTier(container, items, tierClass) {
    container.innerHTML = ''; // Clear previous
    
    // Determine colors based on tier
    let colorHex = '#166534'; // High (Green)
    let bgHex = '#DCFCE7';
    if(tierClass === 'medium') { colorHex = '#B45309'; bgHex = '#FEF3C7'; }
    if(tierClass === 'low') { colorHex = '#B91C1C'; bgHex = '#FEE2E2'; }

    if (items.length === 0) {
        container.innerHTML = `<p class="text-sm text-gray-500 italic">No recommendations found for this tier.</p>`;
        return;
    }

    items.forEach((item, index) => {
        const percentage = Math.min((item.score * 100), 100).toFixed(0);
        const imageUrl = item.imgUrl || 'https://via.placeholder.com/100?text=N/A';
        
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg p-sm shadow-sm flex items-center gap-md border border-outline-variant hover:shadow-md transition-all opacity-0 translate-y-2 cursor-pointer hover:border-primary';
        card.title = `Click to see details for ${item.name}`; 
        card.setAttribute('data-asin', item.asin);
        card.style.transitionDelay = `${index * 100}ms`;

        card.addEventListener('click', () => {
            triggerSearch(item.name, item.asin, categorySelect.value);
            // Scroll back to top smoothly
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        
        card.innerHTML = `
            <div class="w-12 h-12 bg-surface-container rounded-md flex-shrink-0 overflow-hidden flex items-center justify-center border border-outline-variant">
                 <img src="${imageUrl}" class="w-full h-full object-contain" onerror="this.style.display='none'; this.nextElementSibling.style.display='block'">
                 <span class="material-symbols-outlined text-on-surface-variant text-[20px] hidden">inventory_2</span>
            </div>
            <div class="flex-1 min-w-0">
                <p class="font-label-md text-label-md text-on-surface truncate">${item.name}</p>
                <p class="font-body-sm text-body-sm text-on-surface-variant truncate">${item.category}</p>
            </div>
            <div class="text-right">
                <span class="font-h4 text-h4" style="color: ${colorHex}">${item.score.toFixed(2)}</span>
                <div class="w-16 h-1 rounded mt-1 overflow-hidden" style="background-color: ${bgHex}">
                    <div class="h-full w-0 transition-all duration-1000 ease-out" style="background-color: ${colorHex}" data-target-width="${percentage}%"></div>
                </div>
            </div>
        `;
        
        container.appendChild(card);
        
        // Trigger entrance animation
        requestAnimationFrame(() => {
            setTimeout(() => {
                card.classList.remove('opacity-0', 'translate-y-2');
                // Animate bar
                setTimeout(() => {
                    const bar = card.querySelector('.transition-all.ease-out');
                    if (bar) bar.style.width = bar.getAttribute('data-target-width');
                }, 100); 
            }, 50);
        });
    });
}

function renderChart(data) {
    // Combine all and sort by score
    const allItems = [
        ...data.high.map(item => ({...item, tier: 'high', color: '#166534'})),
        ...data.medium.map(item => ({...item, tier: 'medium', color: '#B45309'})),
        ...data.low.map(item => ({...item, tier: 'low', color: '#B91C1C'}))
    ];
    
    // Sort descending
    allItems.sort((a, b) => b.score - a.score);
    
    scoreChart.innerHTML = `
        <div class="absolute w-full border-t border-dashed border-[#166534]/50 z-10" style="bottom: 50%;">
            <span class="absolute right-0 -top-5 text-xs text-[#166534]">0.50</span>
        </div>
        <div class="absolute w-full border-t border-dashed border-[#B45309]/50 z-10" style="bottom: 15%;">
            <span class="absolute right-0 -top-5 text-xs text-[#B45309]">0.15</span>
        </div>
        <div class="absolute inset-0 pt-8 flex items-end justify-between px-2 z-20 gap-1" id="chartBarsContainer"></div>
    `;
    const container = document.getElementById('chartBarsContainer');

    allItems.forEach((item, index) => {
        const percentage = Math.min((item.score * 100), 100).toFixed(1);
        
        const row = document.createElement('div');
        row.className = 'w-full h-full relative group';
        
        row.innerHTML = `
            <div class="absolute bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[12px] rounded-t-sm transition-all duration-1000 ease-out" 
                 style="background-color: ${item.color}; height: 0%;" 
                 data-target-height="${percentage}%"></div>
            
            <!-- Tooltip -->
            <div class="absolute bottom-full mb-2 hidden group-hover:block w-48 p-2 bg-inverse-surface text-inverse-on-surface text-xs rounded shadow-lg z-50">
                ${item.name} (${item.score.toFixed(2)})
            </div>
        `;
        
        container.appendChild(row);
        
        // Animate stagger
        setTimeout(() => {
            const bar = row.firstElementChild;
            if (bar) bar.style.height = bar.getAttribute('data-target-height');
        }, index * 50 + 100);
    });
}

function animatePrecisionRing(targetValue) {
    let current = 0;
    
    let color = '#B91C1C'; // Low (Red)
    if (targetValue >= 80) color = '#166534'; // Green
    else if (targetValue >= 50) color = '#B45309'; // Orange
    
    const interval = setInterval(() => {
        current += 1;
        precisionValue.textContent = `${current}%`;
        precisionRing.style.background = `conic-gradient(${color} ${current * 3.6}deg, #e4e1ee 0deg)`;
        
        if (current >= targetValue) {
            clearInterval(interval);
        }
    }, 20);
}

// --- AI Project Logic ---
function triggerAIProject(goal, apiKey) {
    dashboardContent.classList.add('hidden');
    if (aiDashboardContent) aiDashboardContent.classList.add('hidden');
    selectedProductPanel.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingState.querySelector('p').textContent = "Consulting Gemini AI and matching products...";

    const hostname = window.location.hostname || '127.0.0.1';
    const apiUrl = `http://${hostname}:5000/api/project_recommend`;

    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ goal: goal, api_key: apiKey })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || "API Error"); });
        }
        return response.json();
    })
    .then(data => {
        loadingState.classList.add('hidden');
        renderAIProject(data);
    })
    .catch(error => {
        console.error(error);
        alert("AI Request Failed: " + error.message);
        loadingState.classList.add('hidden');
    });
}

function renderAIProject(data) {
    aiGoalTitle.textContent = `Goal: "${data.goal}"`;
    aiResultsGrid.innerHTML = '';
    
    if (!data.products || data.products.length === 0) {
        aiResultsGrid.innerHTML = `<p class="col-span-full text-center text-on-surface-variant py-8">Could not find matching products for this goal.</p>`;
        aiDashboardContent.classList.remove('hidden');
        return;
    }

    data.products.forEach((product, index) => {
        const imageUrl = product.imgUrl || 'https://via.placeholder.com/150?text=N/A';
        const card = document.createElement('div');
        card.className = 'bg-surface-bright rounded-lg p-md shadow-sm border border-outline-variant flex flex-col gap-sm opacity-0 translate-y-4 hover:shadow-md hover:border-primary transition-all cursor-pointer';
        card.style.transitionDelay = `${index * 100}ms`;

        card.addEventListener('click', () => {
            triggerSearch(product.name, product.asin, categorySelect.value);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        card.innerHTML = `
            <div class="flex justify-between items-start">
                <span class="inline-block bg-primary-container/20 text-primary-container px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider">${product.task_item}</span>
                <span class="font-label-sm text-on-surface-variant flex items-center"><span class="material-symbols-outlined text-[14px] text-tertiary-container mr-1">star</span>${product.rating}</span>
            </div>
            <div class="w-full h-32 bg-surface-container rounded-md flex items-center justify-center overflow-hidden">
                <img src="${imageUrl}" class="max-w-full max-h-full object-contain" onerror="this.style.display='none'; this.nextElementSibling.style.display='block'">
                <span class="material-symbols-outlined text-outline-variant text-[32px] hidden">inventory_2</span>
            </div>
            <div class="flex-1">
                <p class="font-label-md text-on-surface line-clamp-2" title="${product.name}">${product.name}</p>
                <p class="font-body-sm text-on-surface-variant truncate text-xs mt-1" title="${product.category}">${product.category}</p>
            </div>
            <div class="mt-auto pt-sm border-t border-outline-variant flex justify-between items-center">
                <span class="font-h4 text-primary">${product.price}</span>
            </div>
        `;
        
        aiResultsGrid.appendChild(card);
        
        // Trigger animation
        requestAnimationFrame(() => {
            setTimeout(() => {
                card.classList.remove('opacity-0', 'translate-y-4');
            }, 50);
        });
    });

    aiDashboardContent.classList.remove('hidden');
}

// --- Pipeline Scroll Observer ---
function initPipelineObserver() {
    const steps = document.querySelectorAll('.pipeline-step');
    const connectors = document.querySelectorAll('.pipeline-connector');
    
    if (!steps.length) return;
    
    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            steps.forEach((step, index) => {
                setTimeout(() => {
                    step.classList.add('active');
                    if (connectors[index]) {
                        connectors[index].classList.add('active');
                    }
                }, index * 600);
            });
            observer.disconnect();
        }
    }, { threshold: 0.5 });
    
    const pipelineSection = document.querySelector('.pipeline-section');
    if (pipelineSection) {
        observer.observe(pipelineSection);
    }
}

// --- Tab Switching Logic ---
const navDashboard = document.getElementById('navDashboard');
const navCampaigns = document.getElementById('navCampaigns');
const navInsights = document.getElementById('navInsights');

const campaignsView = document.getElementById('campaignsView');
const insightsView = document.getElementById('insightsView');

function switchTab(tab) {
    const mainSections = document.querySelectorAll('main > section');
    
    [navDashboard, navCampaigns, navInsights].forEach(nav => {
        if(nav) {
            nav.classList.remove('active', 'text-primary-container', 'border-b-2', 'border-primary-container', 'pb-1');
            nav.classList.add('text-on-surface-variant');
        }
    });

    mainSections.forEach(sec => sec.classList.add('hidden'));
    dashboardContent.classList.add('hidden');
    if (aiDashboardContent) aiDashboardContent.classList.add('hidden');
    campaignsView.classList.add('hidden');
    insightsView.classList.add('hidden');

    if (tab === 'dashboard') {
        if(navDashboard) {
            navDashboard.classList.add('active', 'text-primary-container', 'border-b-2', 'border-primary-container', 'pb-1');
            navDashboard.classList.remove('text-on-surface-variant');
        }
        mainSections.forEach(sec => sec.classList.remove('hidden'));
    } else if (tab === 'campaigns') {
        if(navCampaigns) {
            navCampaigns.classList.add('active', 'text-primary-container', 'border-b-2', 'border-primary-container', 'pb-1');
            navCampaigns.classList.remove('text-on-surface-variant');
        }
        campaignsView.classList.remove('hidden');
    } else if (tab === 'insights') {
        if(navInsights) {
            navInsights.classList.add('active', 'text-primary-container', 'border-b-2', 'border-primary-container', 'pb-1');
            navInsights.classList.remove('text-on-surface-variant');
        }
        insightsView.classList.remove('hidden');
        loadInsights();
    }
}

if (navDashboard) navDashboard.addEventListener('click', () => switchTab('dashboard'));
if (navCampaigns) navCampaigns.addEventListener('click', () => switchTab('campaigns'));
if (navInsights) navInsights.addEventListener('click', () => switchTab('insights'));

// --- Insights Logic ---
function loadInsights() {
    const totalProductsStat = document.getElementById('totalProductsStat');
    const totalCategoriesStat = document.getElementById('totalCategoriesStat');
    const trendingProductsList = document.getElementById('trendingProductsList');
    
    if(!totalProductsStat) return;
    
    totalProductsStat.textContent = 'Loading...';
    totalCategoriesStat.textContent = 'Loading...';
    trendingProductsList.innerHTML = '<p class="text-on-surface-variant">Fetching trending data...</p>';

    const hostname = window.location.hostname || '127.0.0.1';
    fetch(`http://${hostname}:5000/api/insights`)
        .then(res => res.json())
        .then(data => {
            totalProductsStat.textContent = data.total_products.toLocaleString();
            totalCategoriesStat.textContent = data.total_categories.toLocaleString();
            
            trendingProductsList.innerHTML = '';
            data.trending_products.forEach((product, idx) => {
                const item = document.createElement('div');
                item.className = 'flex items-center gap-md p-sm bg-surface-bright rounded-lg border border-outline-variant';
                item.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-primary-container/20 text-primary-container flex items-center justify-center font-bold text-sm">${idx + 1}</div>
                    <div class="flex-1 min-w-0">
                        <p class="font-label-md text-on-surface truncate">${product.name}</p>
                        <p class="font-body-sm text-on-surface-variant truncate">${product.category}</p>
                    </div>
                    <div class="text-right">
                        <p class="font-label-sm uppercase text-on-surface-variant mb-1">Pop. Score</p>
                        <p class="font-h4 text-tertiary-container">${product.score.toFixed(2)}</p>
                    </div>
                `;
                trendingProductsList.appendChild(item);
            });
        })
        .catch(err => {
            console.error(err);
            totalProductsStat.textContent = 'Error';
            totalCategoriesStat.textContent = 'Error';
            trendingProductsList.innerHTML = '<p class="text-error">Failed to load insights.</p>';
        });
}

// --- Campaigns Logic ---
const generateCampaignBtn = document.getElementById('generateCampaignBtn');
const campaignProductInput = document.getElementById('campaignProductInput');
const campaignTypeSelect = document.getElementById('campaignTypeSelect');
const campaignLoading = document.getElementById('campaignLoading');
const campaignResult = document.getElementById('campaignResult');

if (generateCampaignBtn) {
    generateCampaignBtn.addEventListener('click', () => {
        const productName = campaignProductInput.value.trim();
        const type = campaignTypeSelect.value;
        
        if (!productName) return;
        
        campaignResult.classList.add('hidden');
        campaignLoading.classList.remove('hidden');
        
        const hostname = window.location.hostname || '127.0.0.1';
        fetch(`http://${hostname}:5000/api/generate_campaign`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ product_name: productName, type: type })
        })
        .then(res => res.json())
        .then(data => {
            campaignLoading.classList.add('hidden');
            if (data.error) {
                campaignResult.innerHTML = `<span class="text-error">${data.error}</span>`;
            } else {
                campaignResult.textContent = data.campaign_text;
            }
            campaignResult.classList.remove('hidden');
        })
        .catch(err => {
            campaignLoading.classList.add('hidden');
            campaignResult.innerHTML = `<span class="text-error">Failed to connect to API.</span>`;
            campaignResult.classList.remove('hidden');
        });
    });
}

