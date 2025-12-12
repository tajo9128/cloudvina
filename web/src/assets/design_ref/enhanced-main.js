// Enhanced BioDockify Main JavaScript File
// Advanced functionality with StartPro-inspired effects

document.addEventListener('DOMContentLoaded', function() {
    initializeEnhancedAnimations();
    initializeAdvancedInteractions();
    initializeLiveStats();
    initializeUsageChart();
    initializeEnhancedPricingCalculator();
    initializeScrollAnimations();
    initializeParticleSystem();
});

// Enhanced Animation Initialization
function initializeEnhancedAnimations() {
    // Advanced typewriter effect with gradient text
    if (document.getElementById('typed-text')) {
        new Typed('#typed-text', {
            strings: [
                'for Indian Researchers',
                'Made Simple',
                'Powered by AWS',
                'Affordable & Accessible',
                'Scientific Excellence'
            ],
            typeSpeed: 60,
            backSpeed: 40,
            backDelay: 2500,
            loop: true,
            showCursor: true,
            cursorChar: '|',
            preStringTyped: function(arrayPos, self) {
                // Add glow effect when typing
                self.el.classList.add('typing-glow');
            },
            onStringTyped: function(arrayPos, self) {
                // Remove glow effect after typing
                setTimeout(() => {
                    self.el.classList.remove('typing-glow');
                }, 1000);
            }
        });
    }

    // Initialize live counters
    initializeLiveCounters();
    
    // Initialize advanced particle system
    initializeAdvancedParticles();
}

// Live Statistics Counter
function initializeLiveCounters() {
    const counters = [
        { id: 'jobsCounter', target: 1250000, suffix: 'M+', duration: 3000 },
        { id: 'researchersCounter', target: 52000, suffix: 'K+', duration: 2500 },
        { id: 'uptimeCounter', target: 99.9, suffix: '%', duration: 2000, decimals: 1 }
    ];

    counters.forEach(counter => {
        const element = document.getElementById(counter.id);
        if (element) {
            animateCounter(element, counter.target, counter.suffix, counter.duration, counter.decimals);
        }
    });
}

function animateCounter(element, target, suffix, duration, decimals = 0) {
    let start = 0;
    const increment = target / (duration / 16);
    
    const timer = setInterval(() => {
        start += increment;
        
        if (start >= target) {
            start = target;
            clearInterval(timer);
        }
        
        let displayValue = decimals > 0 ? start.toFixed(decimals) : Math.floor(start);
        
        if (suffix === 'M+' && displayValue >= 1000000) {
            displayValue = (displayValue / 1000000).toFixed(1);
        } else if (suffix === 'K+' && displayValue >= 1000) {
            displayValue = (displayValue / 1000).toFixed(0);
        }
        
        element.textContent = displayValue + suffix;
    }, 16);
}

// Advanced Particle System
function initializeParticleSystem() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;

    // Create advanced molecular particles
    for (let i = 0; i < 25; i++) {
        createAdvancedParticle(particlesContainer, i);
    }
}

function createAdvancedParticle(container, index) {
    const particle = document.createElement('div');
    particle.className = 'molecular-animation';
    
    // Random position
    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = Math.random() * 100 + '%';
    
    // Random size
    const size = Math.random() * 15 + 10;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    
    // Random color from palette
    const colors = ['#0f766e', '#f59e0b', '#f97316', '#84cc16'];
    particle.style.background = colors[Math.floor(Math.random() * colors.length)];
    
    // Random opacity
    particle.style.opacity = Math.random() * 0.7 + 0.3;
    
    container.appendChild(particle);
    
    // Advanced animation
    animateParticle(particle, index);
}

function animateParticle(particle, index) {
    const duration = 8000 + Math.random() * 4000;
    const delay = index * 200;
    
    anime({
        targets: particle,
        translateX: [
            { value: () => anime.random(-150, 150), duration: duration / 2 },
            { value: () => anime.random(-150, 150), duration: duration / 2 }
        ],
        translateY: [
            { value: () => anime.random(-150, 150), duration: duration / 2 },
            { value: () => anime.random(-150, 150), duration: duration / 2 }
        ],
        scale: [
            { value: () => anime.random(0.5, 1.5), duration: duration / 2 },
            { value: () => anime.random(0.5, 1.5), duration: duration / 2 }
        ],
        opacity: [
            { value: () => anime.random(0.3, 0.8), duration: duration / 2 },
            { value: () => anime.random(0.3, 0.8), duration: duration / 2 }
        ],
        rotate: [
            { value: () => anime.random(0, 360), duration: duration / 2 },
            { value: () => anime.random(0, 360), duration: duration / 2 }
        ],
        loop: true,
        easing: 'easeInOutSine',
        delay: delay
    });
}

// Usage Analytics Chart
function initializeUsageChart() {
    const chartElement = document.getElementById('usageChart');
    if (!chartElement) return;

    const chart = echarts.init(chartElement);
    
    const option = {
        title: {
            text: 'Global Usage Trends',
            left: 'center',
            textStyle: {
                color: '#374151',
                fontSize: 18,
                fontWeight: 'bold'
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: '#0f766e'
                }
            }
        },
        legend: {
            data: ['Docking Jobs', 'Active Users', 'Research Papers'],
            bottom: 10
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        },
        xAxis: [
            {
                type: 'category',
                boundaryGap: false,
                data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            }
        ],
        yAxis: [
            {
                type: 'value',
                name: 'Jobs (thousands)',
                position: 'left',
                axisLabel: {
                    formatter: '{value}K'
                }
            },
            {
                type: 'value',
                name: 'Users/Papers',
                position: 'right',
                axisLabel: {
                    formatter: '{value}'
                }
            }
        ],
        series: [
            {
                name: 'Docking Jobs',
                type: 'line',
                smooth: true,
                areaStyle: {
                    opacity: 0.3,
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#0f766e' },
                        { offset: 1, color: 'rgba(15, 118, 110, 0.1)' }
                    ])
                },
                data: [120, 132, 101, 134, 90, 230, 210, 182, 191, 234, 290, 330],
                itemStyle: {
                    color: '#0f766e'
                }
            },
            {
                name: 'Active Users',
                type: 'line',
                yAxisIndex: 1,
                smooth: true,
                data: [8200, 9320, 9010, 9340, 12900, 13300, 13200, 14500, 15200, 16800, 18200, 21500],
                itemStyle: {
                    color: '#f59e0b'
                }
            },
            {
                name: 'Research Papers',
                type: 'line',
                yAxisIndex: 1,
                smooth: true,
                data: [45, 52, 61, 78, 95, 112, 125, 142, 168, 195, 234, 298],
                itemStyle: {
                    color: '#84cc16'
                }
            }
        ]
    };
    
    chart.setOption(option);
    
    // Responsive chart
    window.addEventListener('resize', () => {
        chart.resize();
    });
}

// Enhanced Pricing Calculator
function initializeEnhancedPricingCalculator() {
    const sliders = [
        { id: 'researchersSlider', valueId: 'researchersValue', outputId: 'researchersOutput' },
        { id: 'jobsSlider', valueId: 'jobsValue', outputId: 'jobsOutput' },
        { id: 'salarySlider', valueId: 'salaryValue', outputId: 'salaryOutput' },
        { id: 'timeSlider', valueId: 'timeValue', outputId: 'timeOutput' }
    ];

    sliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        const valueElement = document.getElementById(slider.valueId);
        
        if (element && valueElement) {
            element.addEventListener('input', function() {
                updateSliderValue(this, valueElement);
                calculateEnhancedROI();
            });
            
            // Initialize value
            updateSliderValue(element, valueElement);
        }
    });

    // Initial calculation
    calculateEnhancedROI();
}

function updateSliderValue(slider, valueElement) {
    let value = slider.value;
    let displayValue = value;
    
    if (slider.id === 'salarySlider') {
        displayValue = Math.floor(value / 1000) + 'K';
    } else if (slider.id === 'timeSlider') {
        displayValue = value + 'h';
    }
    
    valueElement.textContent = displayValue;
}

function calculateEnhancedROI() {
    const researchers = parseInt(document.getElementById('researchersSlider').value) || 10;
    const jobsPerResearcher = parseInt(document.getElementById('jobsSlider').value) || 50;
    const salary = parseInt(document.getElementById('salarySlider').value) || 50000;
    const timeSaved = parseFloat(document.getElementById('timeSlider').value) || 2;

    const totalJobs = researchers * jobsPerResearcher;
    const timeSavedMonthly = totalJobs * timeSaved;
    const hourlyRate = salary / (22 * 8);
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

    // Update display with animation
    updateROIValue('totalJobs', totalJobs.toLocaleString());
    updateROIValue('recommendedPlan', recommendedPlan);
    updateROIValue('monthlyCost', '₹' + monthlyCost.toLocaleString());
    updateROIValue('timeSavedMonthly', timeSavedMonthly + ' hours');
    updateROIValue('salarySaved', '₹' + Math.round(salarySaved).toLocaleString());
    updateROIValue('roi', roi.toFixed(0) + '%');
}

function updateROIValue(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        // Add animation class
        element.classList.add('updating');
        
        setTimeout(() => {
            element.textContent = value;
            element.classList.remove('updating');
        }, 150);
    }
}

// Advanced Interactions
function initializeAdvancedInteractions() {
    // Enhanced hover effects for cards
    document.querySelectorAll('.feature-card, .pricing-card, .testimonial-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            anime({
                targets: this,
                scale: 1.02,
                duration: 300,
                easing: 'easeOutQuad'
            });
        });

        card.addEventListener('mouseleave', function() {
            anime({
                targets: this,
                scale: 1,
                duration: 300,
                easing: 'easeOutQuad'
            });
        });
    });

    // Button hover effects
    document.querySelectorAll('.btn-primary, .btn-secondary').forEach(button => {
        button.addEventListener('mouseenter', function() {
            anime({
                targets: this,
                scale: 1.05,
                duration: 200,
                easing: 'easeOutQuad'
            });
        });

        button.addEventListener('mouseleave', function() {
            anime({
                targets: this,
                scale: 1,
                duration: 200,
                easing: 'easeOutQuad'
            });
        });
    });

    // Stats cards hover effects
    document.querySelectorAll('.stats-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            anime({
                targets: this.querySelector('.text-3xl'),
                scale: 1.1,
                color: '#f59e0b',
                duration: 300,
                easing: 'easeOutQuad'
            });
        });

        card.addEventListener('mouseleave', function() {
            anime({
                targets: this.querySelector('.text-3xl'),
                scale: 1,
                color: '#fbbf24',
                duration: 300,
                easing: 'easeOutQuad'
            });
        });
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Scroll Animations
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                
                // Animate feature cards
                if (element.classList.contains('feature-card')) {
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        translateY: [50, 0],
                        scale: [0.9, 1],
                        duration: 800,
                        easing: 'easeOutQuad',
                        delay: Math.random() * 200
                    });
                }

                // Animate pricing cards
                if (element.classList.contains('pricing-card')) {
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        translateY: [30, 0],
                        duration: 600,
                        easing: 'easeOutQuad',
                        delay: Array.from(document.querySelectorAll('.pricing-card')).indexOf(element) * 100
                    });
                }

                // Animate testimonial cards
                if (element.classList.contains('testimonial-card')) {
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        translateX: [index % 2 === 0 ? -50 : 50, 0],
                        duration: 800,
                        easing: 'easeOutQuad',
                        delay: Array.from(document.querySelectorAll('.testimonial-card')).indexOf(element) * 150
                    });
                }

                // Animate statistics
                if (element.classList.contains('stats-card')) {
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        scale: [0.8, 1],
                        duration: 1000,
                        easing: 'easeOutElastic(1, .8)',
                        delay: Math.random() * 300
                    });
                }

                observer.unobserve(element);
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .pricing-card, .testimonial-card, .stats-card').forEach((el, index) => {
        el.style.opacity = '0';
        el.dataset.index = index;
        observer.observe(el);
    });
}

// Live Stats Updates
function initializeLiveStats() {
    // Simulate real-time updates
    setInterval(() => {
        updateLiveStats();
    }, 5000);
}

function updateLiveStats() {
    const jobsElement = document.getElementById('jobsCounter');
    const researchersElement = document.getElementById('researchersCounter');
    
    if (jobsElement) {
        const currentValue = parseInt(jobsElement.textContent.replace(/[^\d]/g, '')) || 1250000;
        const newValue = currentValue + Math.floor(Math.random() * 100);
        animateCounter(jobsElement, newValue, 'M+', 1000);
    }
    
    if (researchersElement) {
        const currentValue = parseInt(researchersElement.textContent.replace(/[^\d]/g, '')) || 52000;
        const newValue = currentValue + Math.floor(Math.random() * 10);
        animateCounter(researchersElement, newValue, 'K+', 1000);
    }
}

// Utility Functions
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

// Performance optimization
const optimizedScrollHandler = debounce(() => {
    // Handle scroll-based animations
    const scrollProgress = window.pageYOffset / (document.body.scrollHeight - window.innerHeight);
    
    // Update navigation based on scroll
    const nav = document.querySelector('nav');
    if (scrollProgress > 0.1) {
        nav.classList.add('scrolled');
    } else {
        nav.classList.remove('scrolled');
    }
}, 16);

window.addEventListener('scroll', optimizedScrollHandler);

// Add loading states for interactive elements
function addLoadingState(element) {
    element.style.opacity = '0.7';
    element.style.pointerEvents = 'none';
}

function removeLoadingState(element) {
    element.style.opacity = '1';
    element.style.pointerEvents = 'auto';
}

// Error handling with user-friendly messages
window.addEventListener('error', function(e) {
    console.warn('BioDockify Enhanced: ', e.message);
    
    // Show user-friendly error message if needed
    if (e.message.includes('undefined') || e.message.includes('null')) {
        // Could implement a toast notification system here
    }
});

// Initialize page transitions
function initializePageTransitions() {
    document.querySelectorAll('a[href$=".html"]').forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            if (href.startsWith('http')) return;
            
            e.preventDefault();
            
            // Create sophisticated loading overlay
            const overlay = document.createElement('div');
            overlay.className = 'fixed inset-0 bg-gradient-to-br from-teal-600 to-teal-700 z-50 flex items-center justify-center';
            overlay.innerHTML = `
                <div class="text-center text-white">
                    <div class="text-6xl mb-6 animate-pulse">🧬</div>
                    <div class="text-2xl font-bold mb-2">Loading BioDockify</div>
                    <div class="text-lg opacity-80">Preparing your molecular docking experience...</div>
                    <div class="mt-8 flex justify-center">
                        <div class="w-16 h-1 bg-white/30 rounded-full overflow-hidden">
                            <div class="h-full bg-white rounded-full animate-pulse" style="width: 100%; animation-duration: 1.5s;"></div>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(overlay);
            
            anime({
                targets: overlay,
                opacity: [0, 1],
                duration: 300,
                easing: 'easeOutQuad',
                complete: () => {
                    setTimeout(() => {
                        window.location.href = href;
                    }, 800);
                }
            });
        });
    });
}

// Initialize page transitions
initializePageTransitions();

// Add CSS for enhanced effects
const style = document.createElement('style');
style.textContent = `
    .typing-glow {
        text-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
    }
    
    .updating {
        animation: pulse 0.3s ease-in-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .scrolled {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .molecular-animation {
        box-shadow: 0 0 10px currentColor;
    }
    
    .feature-card:hover .w-16 {
        transform: scale(1.1) rotate(5deg);
    }
    
    .feature-card .w-16 {
        transition: all 0.3s ease;
    }
`;

document.head.appendChild(style);

// Export functions for external use
window.BioDockifyEnhanced = {
    initializeEnhancedAnimations,
    initializeAdvancedInteractions,
    initializeLiveStats,
    initializeUsageChart,
    calculateEnhancedROI,
    animateCounter,
    initializeParticleSystem
};