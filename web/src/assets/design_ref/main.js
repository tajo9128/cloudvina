// BioDockify Main JavaScript File
// Handles all interactive functionality and animations

document.addEventListener('DOMContentLoaded', function() {
    initializeAnimations();
    initializeInteractions();
    initializePricingCalculator();
    initializeFileConverter();
    initializeScrollAnimations();
});

// Animation Initialization
function initializeAnimations() {
    // Typewriter effect for hero section
    if (document.getElementById('typed-text')) {
        new Typed('#typed-text', {
            strings: [
                'for Indian Researchers',
                'Made Simple',
                'Powered by AWS',
                'Affordable & Accessible'
            ],
            typeSpeed: 50,
            backSpeed: 30,
            backDelay: 2000,
            loop: true,
            showCursor: true,
            cursorChar: '|'
        });
    }

    // Floating molecular particles animation
    createFloatingParticles();

    // Animate cards on scroll
    animateCards();
}

// Create floating molecular particles in hero section
function createFloatingParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'molecule-animation';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 6 + 's';
        particle.style.animationDuration = (Math.random() * 4 + 4) + 's';
        particlesContainer.appendChild(particle);

        // Animate particle movement
        anime({
            targets: particle,
            translateX: [
                { value: Math.random() * 200 - 100, duration: 3000 },
                { value: Math.random() * 200 - 100, duration: 3000 }
            ],
            translateY: [
                { value: Math.random() * 200 - 100, duration: 3000 },
                { value: Math.random() * 200 - 100, duration: 3000 }
            ],
            scale: [
                { value: Math.random() * 0.5 + 0.5, duration: 3000 },
                { value: Math.random() * 0.5 + 0.5, duration: 3000 }
            ],
            opacity: [
                { value: Math.random() * 0.5 + 0.2, duration: 3000 },
                { value: Math.random() * 0.5 + 0.2, duration: 3000 }
            ],
            loop: true,
            easing: 'easeInOutSine'
        });
    }
}

// Initialize interactive elements
function initializeInteractions() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Mobile menu toggle (if implemented)
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');
    
    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');
        });
    }

    // Add hover effects to cards
    document.querySelectorAll('.molecular-card, .pricing-card, .tool-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            anime({
                targets: this,
                scale: 1.02,
                duration: 200,
                easing: 'easeOutQuad'
            });
        });

        card.addEventListener('mouseleave', function() {
            anime({
                targets: this,
                scale: 1,
                duration: 200,
                easing: 'easeOutQuad'
            });
        });
    });
}

// Pricing calculator functionality
function initializePricingCalculator() {
    const billingToggle = document.getElementById('billingToggle');
    const priceElements = document.querySelectorAll('[data-monthly]');
    const pricingText = document.querySelectorAll('[data-pricing]');

    if (billingToggle && priceElements.length > 0) {
        billingToggle.addEventListener('change', function() {
            const isYearly = this.checked;
            
            priceElements.forEach(element => {
                const monthlyPrice = element.getAttribute('data-monthly');
                const yearlyPrice = element.getAttribute('data-yearly');
                
                if (isYearly) {
                    element.textContent = '₹' + yearlyPrice;
                } else {
                    element.textContent = '₹' + monthlyPrice;
                }
            });

            pricingText.forEach(element => {
                if (isYearly) {
                    element.textContent = '/year';
                } else {
                    element.textContent = '/month';
                }
            });
        });
    }

    // ROI Calculator
    const researchersInput = document.getElementById('researchers');
    const jobsInput = document.getElementById('jobsPerResearcher');
    const salaryInput = document.getElementById('salary');
    const timeSavedInput = document.getElementById('timeSaved');

    if (researchersInput && jobsInput && salaryInput && timeSavedInput) {
        [researchersInput, jobsInput, salaryInput, timeSavedInput].forEach(input => {
            input.addEventListener('input', calculateROI);
        });

        // Initial calculation
        calculateROI();
    }
}

// Calculate ROI for institutions
function calculateROI() {
    const researchers = parseInt(document.getElementById('researchers').value) || 10;
    const jobsPerResearcher = parseInt(document.getElementById('jobsPerResearcher').value) || 50;
    const salary = parseInt(document.getElementById('salary').value) || 50000;
    const timeSaved = parseFloat(document.getElementById('timeSaved').value) || 2;

    const totalJobs = researchers * jobsPerResearcher;
    const timeSavedMonthly = totalJobs * timeSaved;
    const hourlyRate = salary / (22 * 8); // Assuming 22 working days, 8 hours per day
    const salarySaved = timeSavedMonthly * hourlyRate;

    // Determine recommended plan
    let recommendedPlan = 'Free';
    let monthlyCost = 0;
    
    if (totalJobs <= 30) {
        recommendedPlan = 'Free';
        monthlyCost = 0;
    } else if (totalJobs <= 100) {
        recommendedPlan = 'Student';
        monthlyCost = 99;
    } else if (totalJobs <= 500) {
        recommendedPlan = 'Researcher';
        monthlyCost = 499;
    } else {
        recommendedPlan = 'Institution';
        monthlyCost = 3999;
    }

    const roi = monthlyCost > 0 ? ((salarySaved - monthlyCost) / monthlyCost * 100) : 0;

    // Update display
    document.getElementById('totalJobs').textContent = totalJobs.toLocaleString();
    document.getElementById('recommendedPlan').textContent = recommendedPlan;
    document.getElementById('monthlyCost').textContent = '₹' + monthlyCost.toLocaleString();
    document.getElementById('timeSavedMonthly').textContent = timeSavedMonthly + ' hours';
    document.getElementById('salarySaved').textContent = '₹' + Math.round(salarySaved).toLocaleString();
    document.getElementById('roi').textContent = roi.toFixed(0) + '%';
}

// File converter functionality
function initializeFileConverter() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const convertBtn = document.getElementById('convertBtn');
    const fileQueue = document.getElementById('fileQueue');

    if (!uploadArea || !fileInput) return;

    // Drag and drop functionality
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleFileUpload(files);
    });

    fileInput.addEventListener('change', function() {
        const files = Array.from(this.files);
        handleFileUpload(files);
    });

    if (convertBtn) {
        convertBtn.addEventListener('click', convertFiles);
    }
}

// Handle file upload
function handleFileUpload(files) {
    const fileQueue = document.getElementById('fileQueue');
    const convertBtn = document.getElementById('convertBtn');
    
    if (!fileQueue) return;

    // Clear existing queue display
    fileQueue.innerHTML = '';

    files.forEach((file, index) => {
        if (file.name.toLowerCase().endsWith('.sdf')) {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="flex items-center justify-between w-full">
                    <div class="flex items-center">
                        <span class="text-green-600 mr-2">📄</span>
                        <span class="font-medium text-gray-900">${file.name}</span>
                        <span class="text-sm text-gray-500 ml-2">(${(file.size / 1024).toFixed(1)} KB)</span>
                    </div>
                    <span class="text-sm text-gray-500">Ready</span>
                </div>
            `;
            fileItem.dataset.file = file.name;
            fileQueue.appendChild(fileItem);
        }
    });

    // Enable convert button if files are valid
    if (convertBtn && fileQueue.children.length > 0) {
        convertBtn.disabled = false;
        convertBtn.textContent = `Convert ${fileQueue.children.length} File(s)`;
    }
}

// Convert files (simulated process)
function convertFiles() {
    const convertBtn = document.getElementById('convertBtn');
    const progressSection = document.getElementById('conversionProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const fileItems = document.querySelectorAll('.file-item');
    
    if (!convertBtn || fileItems.length === 0) return;

    // Disable convert button
    convertBtn.disabled = true;
    convertBtn.textContent = 'Converting...';
    
    // Show progress section
    if (progressSection) {
        progressSection.classList.remove('hidden');
    }

    let completed = 0;
    const total = fileItems.length;

    // Simulate conversion process
    fileItems.forEach((item, index) => {
        setTimeout(() => {
            // Update file status
            const statusSpan = item.querySelector('.text-gray-500');
            statusSpan.className = 'status-converting text-sm font-medium';
            statusSpan.textContent = 'Converting...';

            // Simulate conversion time
            setTimeout(() => {
                completed++;
                
                // Update progress
                const progress = (completed / total) * 100;
                if (progressFill) {
                    progressFill.style.width = progress + '%';
                }
                if (progressText) {
                    progressText.textContent = `${completed}/${total} files`;
                }

                // Update file status
                statusSpan.className = 'status-completed text-sm font-medium';
                statusSpan.textContent = 'Completed';

                // Add download link
                const fileName = item.dataset.file.replace('.sdf', '.pdbqt');
                const downloadLink = document.createElement('a');
                downloadLink.href = '#';
                downloadLink.className = 'text-teal-600 hover:text-teal-700 text-sm font-medium ml-2';
                downloadLink.textContent = 'Download';
                item.querySelector('.flex').appendChild(downloadLink);

                // If all files completed
                if (completed === total) {
                    showDownloadSection();
                    convertBtn.textContent = 'Conversion Complete';
                    
                    setTimeout(() => {
                        convertBtn.disabled = false;
                        convertBtn.textContent = 'Convert More Files';
                    }, 2000);
                }
            }, 1500 + Math.random() * 1000); // Random conversion time
        }, index * 200);
    });
}

// Show download section
function showDownloadSection() {
    const downloadSection = document.getElementById('downloadSection');
    const downloadList = document.getElementById('downloadList');
    const fileItems = document.querySelectorAll('.file-item');

    if (!downloadSection || !downloadList) return;

    // Clear existing download list
    downloadList.innerHTML = '';

    // Add download links for each converted file
    fileItems.forEach(item => {
        const originalFile = item.dataset.file;
        const convertedFile = originalFile.replace('.sdf', '.pdbqt');
        
        const downloadItem = document.createElement('div');
        downloadItem.className = 'flex items-center justify-between p-3 bg-gray-50 rounded-lg';
        downloadItem.innerHTML = `
            <div class="flex items-center">
                <span class="text-teal-600 mr-2">📄</span>
                <span class="font-medium text-gray-900">${convertedFile}</span>
            </div>
            <button class="text-teal-600 hover:text-teal-700 font-medium text-sm">
                Download
            </button>
        `;
        downloadList.appendChild(downloadItem);
    });

    // Show download section
    downloadSection.classList.remove('hidden');

    // Animate download section appearance
    anime({
        targets: downloadSection,
        opacity: [0, 1],
        translateY: [20, 0],
        duration: 500,
        easing: 'easeOutQuad'
    });
}

// Scroll animations
function initializeScrollAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                
                // Animate cards
                if (element.classList.contains('molecular-card') || 
                    element.classList.contains('pricing-card') || 
                    element.classList.contains('tool-card')) {
                    
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        translateY: [30, 0],
                        duration: 600,
                        easing: 'easeOutQuad',
                        delay: Math.random() * 200
                    });
                }

                // Animate sections
                if (element.classList.contains('animate-on-scroll')) {
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        translateY: [50, 0],
                        duration: 800,
                        easing: 'easeOutQuad'
                    });
                }

                observer.unobserve(element);
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.molecular-card, .pricing-card, .tool-card, .animate-on-scroll').forEach(el => {
        el.style.opacity = '0';
        observer.observe(el);
    });
}

// Utility function to animate cards
function animateCards() {
    // Animate hero molecular visualizer
    const visualizer = document.querySelector('.molecular-visualizer');
    if (visualizer) {
        anime({
            targets: visualizer,
            opacity: [0, 1],
            scale: [0.9, 1],
            duration: 1000,
            easing: 'easeOutQuad',
            delay: 500
        });
    }

    // Animate feature icons
    document.querySelectorAll('.feature-icon').forEach((icon, index) => {
        anime({
            targets: icon,
            scale: [0, 1],
            rotate: [0, 360],
            duration: 800,
            easing: 'easeOutElastic(1, .8)',
            delay: 200 * index
        });
    });
}

// Smooth page transitions
function initializePageTransitions() {
    // Add loading animation for page transitions
    document.querySelectorAll('a[href$=".html"]').forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Skip if it's an external link
            if (href.startsWith('http')) return;
            
            e.preventDefault();
            
            // Add loading overlay
            const overlay = document.createElement('div');
            overlay.className = 'fixed inset-0 bg-teal-600 z-50 flex items-center justify-center';
            overlay.innerHTML = `
                <div class="text-center text-white">
                    <div class="text-4xl mb-4">🧬</div>
                    <div class="text-xl font-medium">Loading...</div>
                </div>
            `;
            document.body.appendChild(overlay);

            // Animate overlay
            anime({
                targets: overlay,
                opacity: [0, 1],
                duration: 300,
                easing: 'easeOutQuad',
                complete: () => {
                    setTimeout(() => {
                        window.location.href = href;
                    }, 500);
                }
            });
        });
    });
}

// Initialize page transitions
initializePageTransitions();

// Add loading states for buttons
document.querySelectorAll('button, .btn-primary, .btn-secondary').forEach(button => {
    button.addEventListener('click', function() {
        if (this.disabled) return;
        
        const originalText = this.textContent;
        this.style.opacity = '0.8';
        
        setTimeout(() => {
            this.style.opacity = '1';
        }, 200);
    });
});

// Add tooltip functionality
document.querySelectorAll('[data-tooltip]').forEach(element => {
    element.addEventListener('mouseenter', function() {
        const tooltip = document.createElement('div');
        tooltip.className = 'absolute bg-gray-900 text-white px-2 py-1 rounded text-sm z-50';
        tooltip.textContent = this.getAttribute('data-tooltip');
        tooltip.style.bottom = '100%';
        tooltip.style.left = '50%';
        tooltip.style.transform = 'translateX(-50%)';
        tooltip.style.marginBottom = '8px';
        
        this.style.position = 'relative';
        this.appendChild(tooltip);
    });

    element.addEventListener('mouseleave', function() {
        const tooltip = this.querySelector('.absolute.bg-gray-900');
        if (tooltip) {
            tooltip.remove();
        }
    });
});

// Performance optimization: Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Optimize scroll performance
const optimizedScrollHandler = debounce(() => {
    // Handle scroll-based animations here if needed
}, 16); // ~60fps

window.addEventListener('scroll', optimizedScrollHandler);

// Add error handling for missing elements
window.addEventListener('error', function(e) {
    console.warn('BioDockify: ', e.message);
});

// Export functions for external use
window.BioDockify = {
    initializeAnimations,
    initializeInteractions,
    calculateROI,
    convertFiles,
    handleFileUpload
};