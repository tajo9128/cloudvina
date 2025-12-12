import { Link } from 'react-router-dom'
import { useState } from 'react'
import SEOHelmet from '../components/SEOHelmet'

export default function BlogPage() {
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedCategory, setSelectedCategory] = useState('All')

    const posts = [
        {
            id: 1,
            slug: 'cloud-parallelized-docking-future-virtual-screening',
            title: 'From Days to Hours: Why Cloud-Parallelized Docking is the Future of Virtual Screening',
            excerpt: 'Discover how cloud computing is revolutionizing molecular docking by reducing simulation times from days to hours, enabling high-throughput virtual screening without expensive hardware.',
            date: 'December 1, 2024',
            author: 'Dr. Sarah Chen',
            category: 'Technology',
            readTime: '12 min read',
            image: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=1000',
            keywords: 'cloud docking, parallel docking, virtual screening, autodock vina cloud, high-throughput docking'
        },
        {
            id: 2,
            slug: 'optimizing-autodock-vina-cloud-best-practices',
            title: 'Optimizing AutoDock Vina for the Cloud: Best Practices for Ligand Preparation and Grid Box Definition',
            excerpt: 'Master the technical aspects of cloud-based molecular docking with expert guidance on ligand preparation, receptor optimization, and grid box configuration for maximum accuracy.',
            date: 'November 28, 2024',
            author: 'James Wilson, PhD',
            category: 'Tutorials',
            readTime: '15 min read',
            image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&q=80&w=1000',
            keywords: 'autodock vina tutorial, ligand preparation, grid box definition, PDBQT format, molecular docking optimization'
        },
        {
            id: 3,
            slug: 'democratizing-drug-discovery-small-labs',
            title: 'High-Throughput Screening Without the Supercomputer: Democratizing Drug Discovery for Small Labs',
            excerpt: 'Learn how cloud-based molecular docking platforms are leveling the playing field, enabling small biotech startups and academic labs to compete with big pharma using pay-as-you-go infrastructure.',
            date: 'November 25, 2024',
            author: 'Dr. Michael Rodriguez',
            category: 'Industry Insights',
            readTime: '10 min read',
            image: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&q=80&w=1000',
            keywords: 'drug discovery democratization, small lab drug discovery, affordable docking, biotech startup tools, academic drug discovery'
        },
        {
            id: 4,
            slug: 'integrating-cloud-docking-cadd-pipeline',
            title: 'Integrating Cloud Docking into Your CADD Pipeline: A Step-by-Step Guide',
            excerpt: 'Build an efficient Computer-Aided Drug Design workflow by integrating cloud-based molecular docking with ligand generation, virtual screening, and downstream analysis tools.',
            date: 'November 22, 2024',
            author: 'BioDockify Team',
            category: 'Workflows',
            readTime: '13 min read',
            image: 'https://images.unsplash.com/photo-1581093458791-9f3c3900df4b?auto=format&fit=crop&q=80&w=1000',
            keywords: 'CADD pipeline, drug design workflow, docking integration, API automation, computational drug discovery'
        },
        {
            id: 5,
            slug: 'autodock-vina-vs-other-docking-engines',
            title: 'AutoDock Vina vs. Other Docking Engines: When to Use Cloud-Parallelized Vina for Maximum Efficiency',
            excerpt: 'Comprehensive comparison of AutoDock Vina, Glide, GOLD, and DiffDock. Learn when each docking engine excels and why cloud-parallelized Vina dominates high-throughput virtual screening.',
            date: 'November 18, 2024',
            author: 'Dr. Sarah Chen',
            category: 'Comparisons',
            readTime: '14 min read',
            image: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80&w=1000',
            keywords: 'AutoDock Vina comparison, Glide vs Vina, GOLD docking, DiffDock AI, docking engine selection, virtual screening tools'
        },
        {
            id: 6,
            slug: 'biotech-startup-case-study-cloud-docking',
            title: 'Case Study: How a Biotech Startup Accelerated Lead Optimization by 10x Using Cloud-Based Molecular Docking',
            excerpt: 'Real-world success story of a Boston biotech that used cloud docking to compress drug discovery timelines from years to months, saving $127K and securing Series A funding.',
            date: 'November 15, 2024',
            author: 'BioDockify Team',
            category: 'Case Studies',
            readTime: '11 min read',
            image: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&q=80&w=1000',
            keywords: 'biotech startup case study, drug discovery acceleration, cloud docking success story, lead optimization ROI, computational drug design'
        },
        {
            id: 7,
            slug: 'ai-docking-hybrid-pipelines',
            title: 'AI-Accelerated Docking Meets Classical Methods: The Future of Hybrid Molecular Docking Pipelines',
            excerpt: 'Explore how AI tools like DiffDock and AlphaFold are converging with classical physics-based docking to create powerful hybrid workflows that combine speed, accuracy, and innovation.',
            date: 'November 12, 2024',
            author: 'Dr. Sarah Chen',
            category: 'Emerging Research',
            readTime: '13 min read',
            image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=1000',
            keywords: 'AI molecular docking, hybrid docking pipelines, DiffDock, AlphaFold drug discovery, machine learning docking, future of CADD'
        },
        {
            id: 8,
            slug: 'cost-benefit-cloud-vs-onpremise-docking',
            title: 'Cost-Benefit Analysis: Cloud Docking vs. On-Premise Infrastructure—What Your Lab Actually Saves',
            excerpt: 'Data-driven breakdown of the true total cost of ownership for on-premise clusters vs. cloud docking. Discover why most labs save $80,000-350,000 annually by switching to cloud.',
            date: 'November 8, 2024',
            author: 'BioDockify Team',
            category: 'Cost Analysis',
            readTime: '12 min read',
            image: 'https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?auto=format&fit=crop&q=80&w=1000',
            keywords: 'cloud docking cost analysis, on-premise vs cloud, computational chemistry ROI, drug discovery infrastructure costs, HPC vs cloud'
        }
    ]

    const categories = [
        { name: 'All', count: posts.length },
        { name: 'Technology', count: 1 },
        { name: 'Tutorials', count: 1 },
        { name: 'Industry Insights', count: 1 },
        { name: 'Workflows', count: 1 },
        { name: 'Comparisons', count: 1 },
        { name: 'Case Studies', count: 1 },
        { name: 'Emerging Research', count: 1 },
        { name: 'Cost Analysis', count: 1 }
    ]

    const filteredPosts = posts.filter(post => {
        const matchesSearch = post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            post.excerpt.toLowerCase().includes(searchQuery.toLowerCase())
        const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory
        return matchesSearch && matchesCategory
    })

    const schemaMarkup = {
        "@context": "https://schema.org",
        "@type": "Blog",
        "name": "BioDockify Blog",
        "description": "Expert insights on molecular docking, drug discovery, and computational chemistry",
        "publisher": {
            "@type": "Organization",
            "name": "BioDockify"
        },
        "blogPost": posts.map(post => ({
            "@type": "BlogPosting",
            "headline": post.title,
            "description": post.excerpt,
            "datePublished": post.date,
            "author": {
                "@type": "Person",
                "name": post.author
            },
            "keywords": post.keywords
        }))
    }

    return (
        <>
            <SEOHelmet
                title="Molecular Docking Blog | Drug Discovery Insights & Tutorials - BioDockify"
                description="Expert articles on molecular docking, AutoDock Vina optimization, virtual screening, and computational drug discovery. Learn from industry professionals and researchers."
                keywords="molecular docking blog, autodock vina tutorials, drug discovery articles, computational chemistry, virtual screening guides, CADD workflows"
                canonical="https://biodockify.com/blog"
                schema={schemaMarkup}
            />

            <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
                {/* Hero Section */}
                <section className="bg-gradient-to-r from-slate-900 via-indigo-900 to-slate-900 text-white pt-32 pb-20">
                    <div className="container mx-auto px-4">
                        <div className="max-w-3xl">
                            <h1 className="text-5xl font-bold mb-6">BioDockify Blog</h1>
                            <p className="text-xl text-slate-300 mb-8">
                                Expert insights, tutorials, and industry trends in molecular docking and drug discovery
                            </p>

                            {/* Search Bar */}
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search articles..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full px-6 py-4 rounded-xl text-slate-900 pl-14 focus:outline-none focus:ring-2 focus:ring-primary-500"
                                />
                                <svg className="w-6 h-6 absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                                </svg>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Main Content */}
                <section className="py-16">
                    <div className="container mx-auto px-4">
                        <div className="grid lg:grid-cols-4 gap-8">
                            {/* Sidebar */}
                            <div className="lg:col-span-1">
                                <div className="sticky top-24 space-y-8">
                                    {/* Categories */}
                                    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                                        <h3 className="font-bold text-slate-900 mb-4">Categories</h3>
                                        <div className="space-y-2">
                                            {categories.map(category => (
                                                <button
                                                    key={category.name}
                                                    onClick={() => setSelectedCategory(category.name)}
                                                    className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${selectedCategory === category.name
                                                        ? 'bg-primary-600 text-white'
                                                        : 'hover:bg-slate-100 text-slate-700'
                                                        }`}
                                                >
                                                    {category.name} ({category.count})
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Newsletter */}
                                    <div className="bg-gradient-to-br from-primary-600 to-indigo-700 rounded-xl p-6 text-white">
                                        <h3 className="font-bold mb-2">Stay Updated</h3>
                                        <p className="text-sm text-primary-100 mb-4">Get the latest articles on drug discovery</p>
                                        <input
                                            type="email"
                                            placeholder="your@email.com"
                                            className="w-full px-4 py-2 rounded-lg text-slate-900 mb-3"
                                        />
                                        <button className="w-full bg-white text-primary-600 py-2 rounded-lg font-bold hover:bg-primary-50 transition">
                                            Subscribe
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Blog Posts */}
                            <div className="lg:col-span-3">
                                {filteredPosts.length === 0 ? (
                                    <div className="text-center py-12 text-slate-500">
                                        <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                        </svg>
                                        <p>No articles found matching your search.</p>
                                    </div>
                                ) : (
                                    <div className="grid gap-8">
                                        {filteredPosts.map(post => (
                                            <article key={post.id} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-lg transition-shadow">
                                                <div className="md:flex">
                                                    <div className="md:w-1/3">
                                                        <img src={post.image} alt={post.title} className="w-full h-64 md:h-full object-cover" />
                                                    </div>
                                                    <div className="md:w-2/3 p-8">
                                                        <div className="flex items-center gap-4 mb-4">
                                                            <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-xs font-bold uppercase">
                                                                {post.category}
                                                            </span>
                                                            <span className="text-sm text-slate-500">{post.date}</span>
                                                            <span className="text-sm text-slate-500">• {post.readTime}</span>
                                                        </div>

                                                        <h2 className="text-2xl font-bold text-slate-900 mb-3 hover:text-primary-600 transition">
                                                            <Link to={`/blog/${post.slug}`}>{post.title}</Link>
                                                        </h2>

                                                        <p className="text-slate-600 mb-4 line-clamp-3">
                                                            {post.excerpt}
                                                        </p>

                                                        <div className="flex items-center justify-between">
                                                            <div className="flex items-center gap-3">
                                                                <div className="w-10 h-10 bg-slate-200 rounded-full flex items-center justify-center">
                                                                    <span className="text-sm font-bold text-slate-600">{post.author[0]}</span>
                                                                </div>
                                                                <div>
                                                                    <p className="text-sm font-medium text-slate-900">{post.author}</p>
                                                                </div>
                                                            </div>

                                                            <Link
                                                                to={`/blog/${post.slug}`}
                                                                className="text-primary-600 hover:text-primary-700 font-bold flex items-center gap-2"
                                                            >
                                                                Read More
                                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                                                                </svg>
                                                            </Link>
                                                        </div>
                                                    </div>
                                                </div>
                                            </article>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </>
    )
}
