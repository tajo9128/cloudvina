import { useParams, Link } from 'react-router-dom'
import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import SEOHelmet from '../components/SEOHelmet'

// Import all blog posts
import post1 from './blog/cloud-parallelized-docking-future-virtual-screening.md?raw'
import post2 from './blog/optimizing-autodock-vina-cloud-best-practices.md?raw'
import post3 from './blog/democratizing-drug-discovery-small-labs.md?raw'
import post4 from './blog/integrating-cloud-docking-cadd-pipeline.md?raw'
import post5 from './blog/autodock-vina-vs-other-docking-engines.md?raw'
import post6 from './blog/biotech-startup-case-study-cloud-docking.md?raw'
import post7 from './blog/ai-docking-hybrid-pipelines.md?raw'
import post8 from './blog/cost-benefit-cloud-vs-onpremise-docking.md?raw'

const postsMap = {
    'cloud-parallelized-docking-future-virtual-screening': post1,
    'optimizing-autodock-vina-cloud-best-practices': post2,
    'democratizing-drug-discovery-small-labs': post3,
    'integrating-cloud-docking-cadd-pipeline': post4,
    'autodock-vina-vs-other-docking-engines': post5,
    'biotech-startup-case-study-cloud-docking': post6,
    'ai-docking-hybrid-pipelines': post7,
    'cost-benefit-cloud-vs-onpremise-docking': post8
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
    }
}

export default function BlogPostPage() {
    const { slug } = useParams()
    const content = postsMap[slug]
    const meta = postsMeta[slug]

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

            {/* Hero Image */}
            <div className="h-96 w-full relative">
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/50 to-transparent z-10"></div>
                <img src={meta.image} alt={meta.title} className="w-full h-full object-cover" />
                <div className="absolute bottom-0 left-0 w-full z-20 p-8 md:p-16">
                    <div className="container mx-auto">
                        <div className="max-w-4xl">
                            <span className="px-3 py-1 bg-primary-500 text-white rounded-full text-xs font-bold uppercase mb-4 inline-block">
                                {meta.category}
                            </span>
                            <h1 className="text-3xl md:text-5xl font-bold text-white mb-4 leading-tight">
                                {meta.title}
                            </h1>
                            <div className="flex items-center gap-6 text-slate-300">
                                <span className="flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                                    {meta.author}
                                </span>
                                <span className="flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                    {meta.date}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="container mx-auto px-4 py-16">
                <div className="max-w-3xl mx-auto">
                    <article className="prose prose-lg prose-slate prose-headings:text-slate-900 prose-a:text-primary-600 hover:prose-a:text-primary-500">
                        <ReactMarkdown>{content}</ReactMarkdown>
                    </article>

                    <div className="mt-16 pt-8 border-t border-slate-200">
                        <h3 className="text-2xl font-bold text-slate-900 mb-6">Read Next</h3>
                        <div className="grid md:grid-cols-2 gap-8">
                            {/* Simple logic to show random other posts could go here */}
                            <Link to="/blog" className="block p-6 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors border border-slate-200">
                                <span className="text-primary-600 font-bold text-sm uppercase mb-2 block">Back to Blog</span>
                                <h4 className="text-xl font-bold text-slate-900">Explore more articles →</h4>
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
