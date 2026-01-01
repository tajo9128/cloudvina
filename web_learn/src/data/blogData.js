// Complete blog posts for learn.biodockify.com
// 8 posts from live site + 12 posts from GitHub = 20 total

export const blogPosts = [
    // ========================================
    // FIRST 8: Already live on www.biodockify.com
    // ========================================
    {
        id: 1,
        title: 'From Days to Hours: Why Cloud-Parallelized Docking is the Future of Virtual Screening',
        slug: 'cloud-parallelized-docking-future-virtual-screening',
        excerpt: 'Discover how cloud-parallelized molecular docking reduces screening time from days to hours, enabling faster drug discovery workflows.',
        author: 'Dr. Sarah Chen',
        date: '2024-12-01',
        readTime: '8 min read',
        category: 'Technology',
        tags: ['cloud docking', 'parallel virtual screening', 'AutoDock Vina cloud', 'high-throughput docking'],
        image: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1200&auto=format&fit=crop',
        featured: true
    },
    {
        id: 2,
        title: 'Step-by-Step Guide: Molecular Docking with AutoDock Vina & BioDockify',
        slug: 'step-by-step-guide-molecular-docking-autodock-vina',
        excerpt: 'Complete beginner-friendly tutorial on performing molecular docking using AutoDock Vina through the BioDockify cloud platform.',
        author: 'BioDockify Team',
        date: '2024-11-28',
        readTime: '12 min read',
        category: 'Tutorials',
        tags: ['AutoDock Vina', 'tutorial', 'molecular docking', 'BioDockify', 'step-by-step'],
        image: 'https://images.unsplash.com/photo-1532094349884-543bc11b234d?w=1200&auto=format&fit=crop',
        featured: true
    },
    {
        id: 3,
        title: 'Top 5 Challenges in Molecular Docking and How Cloud Platforms Solve Them',
        slug: 'top-5-challenges-molecular-docking-solutions',
        excerpt: 'Exploring the biggest obstacles in molecular docking workflows and how cloud computing provides practical solutions.',
        author: 'Dr. Sarah Chen',
        date: '2024-11-25',
        readTime: '7 min read',
        category: 'Industry Insights',
        tags: ['molecular docking challenges', 'cloud solutions', 'computational cost', 'data management'],
        image: 'https://images.unsplash.com/photo-1507413245164-6160d8298b31?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 4,
        title: 'Optimizing Virtual Screening Workflows: From Library Preparation to Hit Identification',
        slug: 'optimizing-virtual-screening-workflows',
        excerpt: 'Best practices for designing efficient virtual screening pipelines that maximize hit rates while minimizing false positives.',
        author: 'BioDockify Team',
        date: '2024-11-22',
        readTime: '10 min read',
        category: 'Workflows',
        tags: ['virtual screening', 'workflow optimization', 'library preparation', 'hit identification'],
        image: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?w=1200&auto=format&fit=crop',
        featured: true
    },
    {
        id: 5,
        title: 'AutoDock Vina vs. Other Docking Engines: When to Use Cloud-Parallelized Vina for Maximum Efficiency',
        slug: 'autodock-vina-vs-other-docking-engines',
        excerpt: 'Comprehensive comparison of AutoDock Vina with Glide, GOLD, and DiffDock—and when cloud parallelization gives Vina the edge.',
        author: 'Dr. Sarah Chen',
        date: '2024-11-18',
        readTime: '9 min read',
        category: 'Comparisons',
        tags: ['AutoDock Vina', 'Glide', 'GOLD', 'DiffDock', 'docking engines', 'comparison'],
        image: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 6,
        title: 'Case Study: How a Biotech Startup Accelerated Lead Optimization by 10x Using Cloud-Based Molecular Docking',
        slug: 'biotech-startup-case-study-cloud-docking',
        excerpt: 'Real-world success story of a biotech company saving 6 months and $50k by migrating to cloud docking infrastructure.',
        author: 'BioDockify Team',
        date: '2024-11-15',
        readTime: '6 min read',
        category: 'Case Studies',
        tags: ['case study', 'biotech', 'lead optimization', 'cloud docking', 'drug discovery'],
        image: 'https://images.unsplash.com/photo-1559757175-5700dde675bc?w=1200&auto=format&fit=crop',
        featured: true
    },
    {
        id: 7,
        title: 'AI-Accelerated Docking Meets Classical Methods: The Future of Hybrid Molecular Docking Pipelines',
        slug: 'ai-docking-hybrid-pipelines',
        excerpt: 'Exploring how machine learning models like DiffDock and AlphaFold complement traditional docking engines in modern drug design.',
        author: 'Dr. Sarah Chen',
        date: '2024-11-12',
        readTime: '11 min read',
        category: 'Emerging Research',
        tags: ['AI docking', 'hybrid methods', 'DiffDock', 'AlphaFold', 'classical docking'],
        image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 8,
        title: 'Cost-Benefit Analysis: Cloud Docking vs. On-Premise Infrastructure—What Your Lab Actually Saves',
        slug: 'cost-benefit-cloud-vs-onpremise-docking',
        excerpt: 'Detailed financial breakdown comparing cloud docking services to traditional HPC infrastructure for molecular docking.',
        author: 'BioDockify Team',
        date: '2024-11-08',
        readTime: '8 min read',
        category: 'Cost Analysis',
        tags: ['cost analysis', 'cloud docking', 'on-premise', 'HPC', 'TCO', 'ROI'],
        image: 'https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=1200&auto=format&fit=crop',
        featured: false
    },

    // ========================================
    // NEXT 12: From GitHub markdown files
    // ========================================
    {
        id: 9,
        title: 'ADMET Prediction and Filtering: Pre-Screening Compounds Before Molecular Docking to Maximize Lead Quality',
        slug: 'admet-prediction-filtering',
        excerpt: 'Learn how ADMET prediction saves time and improves drug discovery success. Discover practical filtering thresholds for absorption, distribution, metabolism, excretion, and toxicity screening.',
        author: 'BioDockify Team',
        date: '2024-12-06',
        readTime: '9 min read',
        category: 'Drug Discovery',
        tags: ['ADMET', 'drug discovery', 'molecular docking', 'pharmacokinetics', 'toxicity prediction', 'Lipinski'],
        image: 'https://images.unsplash.com/photo-1579154204601-01588f351e67?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 10,
        title: 'Cross-Docking Validation and Ensemble Docking: Improving Computational Predictions Through Multi-Target Analysis',
        slug: 'cross-docking-ensemble',
        excerpt: 'Master cross-docking validation and ensemble docking techniques for improved molecular docking predictions. Learn workflows for multi-target screening and polypharmacology.',
        author: 'BioDockify Team',
        date: '2024-12-06',
        readTime: '12 min read',
        category: 'Drug Discovery',
        tags: ['cross-docking', 'ensemble docking', 'molecular docking', 'validation', 'RMSD', 'polypharmacology'],
        image: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 11,
        title: 'High-Throughput Screening Without the Supercomputer: Democratizing Drug Discovery for Small Labs',
        slug: 'democratizing-drug-discovery-small-labs',
        excerpt: 'Cloud-based molecular docking levels the playing field for academic labs and startups. Learn how to run professional virtual screening campaigns on a budget.',
        author: 'Dr. Michael Rodriguez',
        date: '2024-11-25',
        readTime: '10 min read',
        category: 'Industry Insights',
        tags: ['democratizing drug discovery', 'affordable molecular docking', 'small lab tools', 'biotech startup'],
        image: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1200&auto=format&fit=crop',
        featured: true
    },
    {
        id: 12,
        title: 'High-Throughput Virtual Screening vs. Fragment-Based Drug Discovery: Computational Strategies for Different Research Scenarios',
        slug: 'htvs-vs-fbdd',
        excerpt: 'Compare HTVS and FBDD approaches for drug discovery. Learn when to use each computational strategy, workflow differences, and how to optimize your research pipeline.',
        author: 'BioDockify Team',
        date: '2024-12-06',
        readTime: '11 min read',
        category: 'Drug Discovery',
        tags: ['virtual screening', 'fragment-based drug discovery', 'HTVS', 'FBDD', 'hit identification'],
        image: 'https://images.unsplash.com/photo-1581093588401-fbb62a02f120?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 13,
        title: 'Integrating Cloud Docking into Your Computational Drug Discovery Pipeline',
        slug: 'integrating-cloud-docking-cadd-pipeline',
        excerpt: 'Step-by-step guide to integrating cloud-based molecular docking into existing CADD workflows using APIs and automation.',
        author: 'BioDockify Team',
        date: '2024-12-05',
        readTime: '10 min read',
        category: 'Tutorials',
        tags: ['cloud docking', 'CADD pipeline', 'automation', 'API integration', 'workflow'],
        image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 14,
        title: 'Molecular Dynamics vs. Molecular Docking: When to Use Each Computational Technique',
        slug: 'md-vs-docking',
        excerpt: 'Understand the differences between MD simulations and molecular docking. Learn which technique to use for specific drug discovery challenges.',
        author: 'Dr. Sarah Chen',
        date: '2024-12-04',
        readTime: '9 min read',
        category: 'Comparisons',
        tags: ['molecular dynamics', 'molecular docking', 'MD simulation', 'computational chemistry', 'drug discovery'],
        image: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 15,
        title: 'ML-Enhanced Binding Affinity Predictions: Beyond Traditional Scoring Functions',
        slug: 'ml-binding-affinity',
        excerpt: 'Discover how machine learning models enhance binding affinity predictions and complement traditional docking scoring functions.',
        author: 'Dr. Sarah Chen',
        date: '2024-12-03',
        readTime: '10 min read',
        category: 'Emerging Research',
        tags: ['machine learning', 'binding affinity', 'scoring functions', 'AI', 'drug discovery'],
        image: 'https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 16,
        title: 'Pharmacophore-Guided Molecular Docking for Targeted Drug Discovery',
        slug: 'pharmacophore-guided-docking',
        excerpt: 'Learn how to use pharmacophore models to guide molecular docking and improve virtual screening hit rates.',
        author: 'BioDockify Team',
        date: '2024-12-02',
        readTime: '11 min read',
        category: 'Drug Discovery',
        tags: ['pharmacophore', 'molecular docking', 'drug design', 'virtual screening', 'ligand-based'],
        image: 'https://images.unsplash.com/photo-1576671081837-49000212a370?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 17,
        title: 'From Protein Structure to Drug Binding: A Step-by-Step Guide to Protein Preparation for Molecular Docking',
        slug: 'protein-preparation-guide',
        excerpt: 'Learn the essential protein preparation workflow for molecular docking, including PDB file processing, protonation states, and charge assignment.',
        author: 'BioDockify Team',
        date: '2024-12-06',
        readTime: '8 min read',
        category: 'Drug Discovery',
        tags: ['protein preparation', 'molecular docking', 'PDB file', 'AutoDock Vina', 'protonation states'],
        image: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 18,
        title: 'Research to Clinical Translation: Bridging the Gap with Computational Drug Design',
        slug: 'research-to-clinical',
        excerpt: 'Explore how computational drug discovery accelerates the journey from research to clinical trials and regulatory approval.',
        author: 'Dr. Sarah Chen',
        date: '2024-12-01',
        readTime: '10 min read',
        category: 'Industry Insights',
        tags: ['clinical trials', 'drug development', 'computational chemistry', 'translational research', 'regulatory'],
        image: 'https://images.unsplash.com/photo-1582719471384-894fbb16e074?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 19,
        title: 'Scoring Functions in Molecular Docking: Which One Should You Choose for Your Drug Discovery Project?',
        slug: 'scoring-functions-guide',
        excerpt: 'Comprehensive guide to molecular docking scoring functions including AutoDock Vina, Glide, PLANTS, and GOLD. Learn how to select the right scoring function.',
        author: 'BioDockify Team',
        date: '2024-12-06',
        readTime: '10 min read',
        category: 'Drug Discovery',
        tags: ['scoring functions', 'molecular docking', 'AutoDock Vina', 'Glide', 'GOLD', 'binding affinity'],
        image: 'https://images.unsplash.com/photo-1518331368925-fd8d678778e0?w=1200&auto=format&fit=crop',
        featured: false
    },
    {
        id: 20,
        title: 'Virtual Screening for Natural Product Drug Discovery: Strategies and Success Stories',
        slug: 'virtual-screening-natural-products',
        excerpt: 'Discover computational strategies for virtual screening of natural products and phytochemicals in drug discovery campaigns.',
        author: 'BioDockify Team',
        date: '2024-11-30',
        readTime: '9 min read',
        category: 'Drug Discovery',
        tags: ['natural products', 'virtual screening', 'phytochemicals', 'drug discovery', 'traditional medicine'],
        image: 'https://images.unsplash.com/photo-1603796846097-bee99e4a601f?w=1200&auto=format&fit=crop',
        featured: false
    }
];

// Category counts
export const categoryCounts = {
    // New Categories
    'Bioinformatics': 0,
    'Molecular Docking': 0,
    'Molecular Dynamics': 0,
    'AI in Drug Discovery': 0,
    'Workflows': 1,
    'Tutorials': 2,
    'Case Studies': 1,
    'Phytochemical Research': 0,
    'Comparisons': 2,
    'Platform Updates': 0,
    'Beginner’s Corner': 0,

    // Legacy Categories (Preserved for existing posts)
    'Technology': 1,
    'Industry Insights': 3,
    'Emerging Research': 2,
    'Cost Analysis': 1,
    'Drug Discovery': 7
};

export const categories = categoryCounts;

// Get featured posts
export const getFeaturedPosts = () => blogPosts.filter(post => post.featured);

// Get posts by category
export const getPostsByCategory = (category) =>
    blogPosts.filter(post => post.category === category);

// Search posts
export const searchPosts = (query) => {
    const lowerQuery = query.toLowerCase();
    return blogPosts.filter(post =>
        post.title.toLowerCase().includes(lowerQuery) ||
        post.excerpt.toLowerCase().includes(lowerQuery) ||
        post.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
    );
};

export default blogPosts;
