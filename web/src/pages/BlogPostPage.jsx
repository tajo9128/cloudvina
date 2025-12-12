import { useParams, Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import SEOHelmet from '../components/SEOHelmet'

// Import all blog posts
// Batch 1 (Initial 8)
import post1 from './blog/cloud-parallelized-docking-future-virtual-screening.md?raw'
import post2 from './blog/optimizing-autodock-vina-cloud-best-practices.md?raw'
import post3 from './blog/democratizing-drug-discovery-small-labs.md?raw'
import post4 from './blog/integrating-cloud-docking-cadd-pipeline.md?raw'
import post5 from './blog/autodock-vina-vs-other-docking-engines.md?raw'
import post6 from './blog/biotech-startup-case-study-cloud-docking.md?raw'
import post7 from './blog/ai-docking-hybrid-pipelines.md?raw'
import post8 from './blog/cost-benefit-cloud-vs-onpremise-docking.md?raw'

// Batch 2 (New 12)
import post9 from './blog/protein-preparation-guide.md?raw'
import post10 from './blog/scoring-functions-guide.md?raw'
import post11 from './blog/virtual-screening-natural-products.md?raw'
import post12 from './blog/md-vs-docking.md?raw'
import post13 from './blog/ml-binding-affinity.md?raw'
import post14 from './blog/htvs-vs-fbdd.md?raw'
import post15 from './blog/pharmacophore-guided-docking.md?raw'
import post16 from './blog/cross-docking-ensemble.md?raw'
import post17 from './blog/admet-prediction-filtering.md?raw'
import post18 from './blog/sar-docking-analysis.md?raw'
import post19 from './blog/selectivity-kinase-docking.md?raw'
import post20 from './blog/research-to-clinical.md?raw'

const postsMap = {
    'cloud-parallelized-docking-future-virtual-screening': post1,
    'optimizing-autodock-vina-cloud-best-practices': post2,
    'democratizing-drug-discovery-small-labs': post3,
    'integrating-cloud-docking-cadd-pipeline': post4,
    'autodock-vina-vs-other-docking-engines': post5,
    'biotech-startup-case-study-cloud-docking': post6,
    'ai-docking-hybrid-pipelines': post7,
    'cost-benefit-cloud-vs-onpremise-docking': post8,
    'protein-preparation-guide': post9,
    'scoring-functions-guide': post10,
    'virtual-screening-natural-products': post11,
    'md-vs-docking': post12,
    'ml-binding-affinity': post13,
    'htvs-vs-fbdd': post14,
    'pharmacophore-guided-docking': post15,
    'cross-docking-ensemble': post16,
    'admet-prediction-filtering': post17,
    'sar-docking-analysis': post18,
    'selectivity-kinase-docking': post19,
    'research-to-clinical': post20
}

const postsMeta = {
    'cloud-parallelized-docking-future-virtual-screening': {
        title: 'From Days to Hours: Why Cloud-Parallelized Docking is the Future of Virtual Screening',
        excerpt: 'Discover how cloud computing is revolutionizing molecular docking by reducing simulation times from days to hours.',
        date: 'December 1, 2024',
        author: 'Dr. Sarah Chen',
        category: 'Technology',
        image: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=1000'
    },
    'optimizing-autodock-vina-cloud-best-practices': {
        title: 'Optimizing AutoDock Vina for the Cloud: Best Practices',
        excerpt: 'Master the technical aspects of cloud-based molecular docking with expert guidance.',
        date: 'November 28, 2024',
        author: 'James Wilson, PhD',
        category: 'Tutorials',
        image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&q=80&w=1000'
    },
    'democratizing-drug-discovery-small-labs': {
        title: 'Democratizing Drug Discovery for Small Labs',
        excerpt: 'How cloud-based platforms are leveling the playing field for startups and academic labs.',
        date: 'November 25, 2024',
        author: 'Dr. Michael Rodriguez',
        category: 'Industry Insights',
        image: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&q=80&w=1000'
    },
    'integrating-cloud-docking-cadd-pipeline': {
        title: 'Integrating Cloud Docking into Your CADD Pipeline',
        excerpt: 'Build an efficient workflow by integrating cloud docking with other tools.',
        date: 'November 22, 2024',
        author: 'BioDockify Team',
        category: 'Workflows',
        image: 'https://images.unsplash.com/photo-1581093458791-9f3c3900df4b?auto=format&fit=crop&q=80&w=1000'
    },
    'autodock-vina-vs-other-docking-engines': {
        title: 'AutoDock Vina vs. Other Docking Engines',
        excerpt: 'Comparison of Vina, Glide, GOLD, and DiffDock for high-throughput screening.',
        date: 'November 18, 2024',
        author: 'Dr. Sarah Chen',
        category: 'Comparisons',
        image: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80&w=1000'
    },
    'biotech-startup-case-study-cloud-docking': {
        title: 'Case Study: Biotech Startup Accelerates Lead Optimization',
        excerpt: 'How a startup saved $127K and secured funding using cloud docking.',
        date: 'November 15, 2024',
        author: 'BioDockify Team',
        category: 'Case Studies',
        image: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&q=80&w=1000'
    },
    'ai-docking-hybrid-pipelines': {
        title: 'AI-Accelerated Docking Meets Classical Methods',
        excerpt: 'The future of hybrid molecular docking pipelines using AI and physics.',
        date: 'November 12, 2024',
        author: 'Dr. Sarah Chen',
        category: 'Emerging Research',
        image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=1000'
    },
    'cost-benefit-cloud-vs-onpremise-docking': {
        title: 'Cost-Benefit Analysis: Cloud vs. On-Premise Docking',
        excerpt: 'Why labs save $80k-350k annually by switching to cloud infrastructure.',
        date: 'November 8, 2024',
        author: 'BioDockify Team',
        category: 'Cost Analysis',
        image: 'https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?auto=format&fit=crop&q=80&w=1000'
    },
    'protein-preparation-guide': {
        title: 'Protein Structure Preparation for Molecular Docking',
        excerpt: 'Essential guide to preparing PDB files, handling protonation states, and fixing missing atoms.',
        date: 'December 2, 2024',
        author: 'Dr. Sarah Chen',
        category: 'Tutorials',
        image: '/blog/images/protein-preparation-hero.png'
    },
    'scoring-functions-guide': {
        title: 'Understanding Scoring Functions in Molecular Docking',
        excerpt: 'A deep dive into how docking programs rank binding poses and estimate affinity.',
        date: 'December 2, 2024',
        author: 'James Wilson, PhD',
        category: 'Technology',
        image: '/blog/images/scoring-functions-hero.png'
    },
    'virtual-screening-natural-products': {
        title: 'Virtual Screening for Natural Product Drug Discovery',
        excerpt: 'Strategies for screening large natural product databases effectively.',
        date: 'December 3, 2024',
        author: 'Dr. Michael Rodriguez',
        category: 'Research',
        image: '/blog/images/virtual-screening-hero.png'
    },
    'md-vs-docking': {
        title: 'Molecular Dynamics vs. Molecular Docking',
        excerpt: 'Comparing static and dynamic approaches to understanding molecular interactions.',
        date: 'December 3, 2024',
        author: 'BioDockify Team',
        category: 'Comparisons',
        image: '/blog/images/md-vs-docking-hero.png'
    },
    'ml-binding-affinity': {
        title: 'ML-Enhanced Binding Affinity Predictions',
        excerpt: 'Integrating machine learning models to improve scoring accuracy beyond classical functions.',
        date: 'December 4, 2024',
        author: 'Dr. Sarah Chen',
        category: 'Technology',
        image: '/blog/images/ml-binding-affinity-hero.png'
    },
    'htvs-vs-fbdd': {
        title: 'High-Throughput Virtual Screening vs. Fragment-Based Design',
        excerpt: 'Comparing top-down HTVS with bottom-up fragment growing strategies.',
        date: 'December 4, 2024',
        author: 'James Wilson, PhD',
        category: 'Comparisons',
        image: '/blog/images/htvs-vs-fbdd-hero.png'
    },
    'pharmacophore-guided-docking': {
        title: 'Pharmacophore-Guided Molecular Docking',
        excerpt: 'Using spatial constraints and essential features to filter docking poses for better results.',
        date: 'December 5, 2024',
        author: 'Dr. Michael Rodriguez',
        category: 'Tutorials',
        image: '/blog/images/pharmacophore-docking-hero.png'
    },
    'cross-docking-ensemble': {
        title: 'Cross-Docking and Ensemble Docking',
        excerpt: 'Accounting for protein flexibility by docking against multiple receptor conformations.',
        date: 'December 5, 2024',
        author: 'BioDockify Team',
        category: 'Advanced Methods',
        image: '/blog/images/cross-docking-ensemble-hero.png'
    },
    'admet-prediction-filtering': {
        title: 'ADMET Prediction and Filtering',
        excerpt: 'Pre-screening compounds for absorption, distribution, metabolism, excretion, and toxicity.',
        date: 'December 6, 2024',
        author: 'Dr. Sarah Chen',
        category: 'Drug Discovery',
        image: '/blog/images/admet-filtering-hero.png'
    },
    'sar-docking-analysis': {
        title: 'SAR Analysis and Molecular Docking',
        excerpt: 'Connecting computational binding predictions to experimental Structure-Activity Relationships.',
        date: 'December 6, 2024',
        author: 'James Wilson, PhD',
        category: 'Drug Discovery',
        image: '/blog/images/sar-docking-hero.png'
    },
    'selectivity-kinase-docking': {
        title: 'Target Selectivity and Kinase Selectivity',
        excerpt: 'Designing selective inhibitors to hit your target while avoiding off-target toxicity.',
        date: 'December 6, 2024',
        author: 'BioDockify Team',
        category: 'Advanced Methods',
        image: '/blog/images/selectivity-docking-hero.png'
    },
    'research-to-clinical': {
        title: 'From Published Research to Clinical Trials',
        excerpt: 'How computational drug discovery accelerates the translational path from bench to bedside.',
        date: 'December 6, 2024',
        author: 'Dr. Michael Rodriguez',
        category: 'Industry Insights',
        image: '/blog/images/research-clinical-hero.png'
    }
}

export default function BlogPostPage() {
    const { slug } = useParams()
    const content = postsMap[slug]
    const meta = postsMeta[slug]

    // Scroll to top on load
    useEffect(() => {
        window.scrollTo(0, 0)
    }, [slug])

    if (!content || !meta) {
        return (
            <div className="min-h-screen bg-slate-50 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-4xl font-bold text-slate-900 mb-4">Article Not Found</h1>
                    <Link to="/blog" className="text-primary-600 hover:underline">Return to Blog</Link>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-white">
            <SEOHelmet
                title={`${meta.title} | BioDockify Blog`}
                description={meta.excerpt}
                keywords={`molecular docking, ${meta.category.toLowerCase()}, drug discovery, ${slug.replace(/-/g, ' ')}`}
                canonical={`https://biodockify.com/blog/${slug}`}
                schema={{
                    "@context": "https://schema.org",
                    "@type": "BlogPosting",
                    "headline": meta.title,
                    "image": meta.image,
                    "author": {
                        "@type": "Person",
                        "name": meta.author
                    },
                    "publisher": {
                        "@type": "Organization",
                        "name": "BioDockify",
                        "logo": {
                            "@type": "ImageObject",
                            "url": "https://biodockify.com/logo.png"
                        }
                    },
                    "datePublished": meta.date
                }}
            />

            {/* Antler-style Hero Header with Overlay */}
            <div className="relative w-full h-[500px] bg-slate-900 overflow-hidden">
                {/* Background Image with Gradient Overlay */}
                <div className="absolute inset-0">
                    <img
                        src={meta.image}
                        alt={meta.title}
                        className="w-full h-full object-cover opacity-60"
                        onError={(e) => { e.target.src = 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&q=80&w=1000' }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-slate-900 to-transparent/30"></div>
                </div>

                {/* Content Overlay */}
                <div className="absolute bottom-0 left-0 w-full p-8 pb-16">
                    <div className="container mx-auto max-w-6xl">
                        <div className="max-w-4xl">
                            {/* Category Label */}
                            <span className="inline-block px-4 py-1.5 bg-primary-600 text-white rounded-full text-xs font-bold uppercase tracking-wider mb-6">
                                {meta.category}
                            </span>

                            {/* Title */}
                            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
                                {meta.title}
                            </h1>

                            {/* Meta Info */}
                            <div className="flex flex-wrap items-center gap-6 text-slate-300 text-sm md:text-base">
                                <span className="flex items-center gap-2">
                                    <div className="w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center text-white font-bold">
                                        {meta.author[0]}
                                    </div>
                                    <span className="font-medium text-white">{meta.author}</span>
                                </span>
                                <span className="flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                    {meta.date}
                                </span>
                                <span className="flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                    10-15 min read
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Layout with Sidebar */}
            <div className="container mx-auto px-4 py-16 max-w-6xl">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                    {/* Left Column: Social Share (Sticky) - Optional or hidden on mobile */}
                    <div className="hidden lg:block lg:col-span-1">
                        <div className="sticky top-32 flex flex-col gap-4 text-slate-400">
                            <button className="w-10 h-10 rounded-full border border-slate-200 flex items-center justify-center hover:bg-blue-500 hover:text-white hover:border-blue-500 transition-all" title="Share on Twitter">
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M23 3a10.9 10.9 0 0 1-3.14 1.53 4.48 4.48 0 0 0-7.86 3v1A10.66 10.66 0 0 1 3 4s-4 9 5 13a11.64 11.64 0 0 1-7 2c9 5 20 0 20-11.5a4.5 4.5 0 0 0-.08-.83A7.72 7.72 0 0 0 23 3z"></path></svg>
                            </button>
                            <button className="w-10 h-10 rounded-full border border-slate-200 flex items-center justify-center hover:bg-blue-700 hover:text-white hover:border-blue-700 transition-all" title="Share on LinkedIn">
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
                            </button>
                            <button className="w-10 h-10 rounded-full border border-slate-200 flex items-center justify-center hover:bg-blue-600 hover:text-white hover:border-blue-600 transition-all" title="Share on Facebook">
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path></svg>
                            </button>
                        </div>
                    </div>

                    {/* Middle Column: Article Content */}
                    <div className="lg:col-span-8">
                        <article className="prose prose-lg prose-slate max-w-none 
                            prose-headings:font-bold prose-headings:text-slate-900 
                            prose-h1:text-4xl prose-h2:text-3xl prose-h2:mt-12 prose-h2:mb-6 prose-h2:pb-4 prose-h2:border-b prose-h2:border-slate-100
                            prose-h3:text-2xl prose-h3:mt-8 prose-h3:mb-4
                            prose-p:text-slate-600 prose-p:leading-relaxed prose-p:mb-6
                            prose-a:text-primary-600 prose-a:no-underline hover:prose-a:underline hover:prose-a:text-primary-700
                            prose-blockquote:border-l-4 prose-blockquote:border-primary-500 prose-blockquote:bg-slate-50 prose-blockquote:p-6 prose-blockquote:rounded-r-lg prose-blockquote:not-italic prose-blockquote:text-slate-700
                            prose-img:rounded-xl prose-img:shadow-lg prose-img:my-8
                            prose-code:text-primary-600 prose-code:bg-primary-50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none
                            prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-pre:rounded-xl prose-pre:shadow-lg prose-pre:p-6
                            prose-li:text-slate-600
                            ">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
                        </article>

                        {/* Author Box */}
                        <div className="mt-16 bg-slate-50 rounded-2xl p-8 border border-slate-100 flex flex-col md:flex-row gap-6 items-center md:items-start text-center md:text-left">
                            <div className="w-20 h-20 rounded-full bg-white shadow-md p-1 flex-shrink-0">
                                <div className="w-full h-full rounded-full bg-primary-100 flex items-center justify-center text-2xl font-bold text-primary-600">
                                    {meta.author[0]}
                                </div>
                            </div>
                            <div>
                                <h3 className="text-xl font-bold text-slate-900 mb-2">About {meta.author}</h3>
                                <p className="text-slate-600 mb-4">
                                    Expert contributor to the BioDockify scientific blog. Specializing in computational chemistry, drug discovery, and molecular modeling technologies.
                                </p>
                                <Link to="/blog" className="text-primary-600 hover:text-primary-700 font-semibold text-sm">
                                    View all posts by {meta.author} →
                                </Link>
                            </div>
                        </div>

                        {/* Navigation Buttons (Prev/Next) - keeping it simple for now */}
                        <div className="mt-12 pt-8 border-t border-slate-200">
                            <Link to="/blog" className="inline-flex items-center gap-2 text-slate-600 hover:text-primary-600 font-medium transition-colors">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                                Back to All Articles
                            </Link>
                        </div>
                    </div>

                    {/* Right Column: Sidebar */}
                    <div className="lg:col-span-3 space-y-8">
                        {/* Search Widget */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4 text-lg">Search</h3>
                            <div className="relative">
                                <input type="text" placeholder="Search..." className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all text-sm" />
                                <svg className="w-4 h-4 absolute right-4 top-1/2 transform -translate-y-1/2 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                            </div>
                        </div>

                        {/* Recent Posts Widget */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-6 text-lg border-b border-slate-100 pb-2">Recent Articles</h3>
                            <div className="space-y-6">
                                {['protein-preparation-guide', 'admet-prediction-filtering', 'sar-docking-analysis'].map(key => (
                                    key !== slug && postsMeta[key] && (
                                        <Link to={`/blog/${key}`} key={key} className="block group">
                                            <h4 className="font-bold text-slate-800 text-sm group-hover:text-primary-600 transition-colors mb-2 line-clamp-2">
                                                {postsMeta[key].title}
                                            </h4>
                                            <span className="text-xs text-slate-500">{postsMeta[key].date}</span>
                                        </Link>
                                    )
                                ))}
                            </div>
                        </div>

                        {/* Tags/Categories Cloud */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-6 text-lg border-b border-slate-100 pb-2">Topics</h3>
                            <div className="flex flex-wrap gap-2">
                                {['Molecular Docking', 'Drug Discovery', 'Tutorials', 'Case Studies', 'AI', 'Cloud Computing', 'ADMET', 'Virtual Screening'].map(tag => (
                                    <span key={tag} className="px-3 py-1 bg-slate-50 text-slate-600 text-xs rounded-full border border-slate-100 hover:bg-primary-50 hover:text-primary-600 hover:border-primary-100 transition-colors cursor-pointer">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>

                        {/* CTA Banner */}
                        <div className="bg-gradient-to-br from-indigo-900 to-slate-900 rounded-xl p-8 text-center text-white">
                            <h3 className="font-bold text-xl mb-3">Start Docking Today</h3>
                            <p className="text-sm text-slate-300 mb-6">BioDockify makes drug discovery accessible to everyone.</p>
                            <Link to="/login" className="inline-block w-full py-3 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-lg transition-colors">
                                Try for Free
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
