import { Link } from 'react-router-dom'
import { useState } from 'react'

export default function BlogPage() {
    const [searchQuery, setSearchQuery] = useState('')

    const posts = [
        {
            id: 1,
            title: 'Accelerating Drug Discovery with Cloud Computing',
            excerpt: 'How cloud-native molecular docking is changing the landscape of pharmaceutical research.',
            date: 'Nov 23, 2025',
            author: 'Dr. Sarah Chen',
            category: 'Technology',
            image: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&q=80&w=1000'
        },
        {
            id: 2,
            title: 'Understanding AutoDock Vina Scoring Functions',
            excerpt: 'A deep dive into how Vina calculates binding affinity and what it means for your results.',
            date: 'Nov 20, 2025',
            author: 'James Wilson',
            category: 'Science',
            image: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&q=80&w=1000'
        },
        {
            id: 3,
            title: 'New Feature: SDF to PDBQT Converter',
            excerpt: 'We have just released a free tool to help you prepare your ligands for docking.',
            date: 'Nov 15, 2025',
            author: 'BioDockify Team',
            category: 'Updates',
            image: 'https://images.unsplash.com/photo-1581093458791-9f3c3900df4b?auto=format&fit=crop&q=80&w=1000'
        },
        {
            id: 4,
            title: 'Best Practices for Ligand Preparation',
            excerpt: 'Ensure your docking results are accurate by following these essential ligand prep steps.',
            date: 'Nov 10, 2025',
            author: 'Dr. Sarah Chen',
            category: 'Tutorials',
            image: 'https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80&w=1000'
        }
    ]

    const categories = [
        { name: 'Technology', count: 12 },
        { name: 'Science', count: 8 },
        { name: 'Updates', count: 5 },
        { name: 'Tutorials', count: 15 },
        { name: 'Case Studies', count: 3 }
    ]

    const tags = ['Molecular Docking', 'AutoDock Vina', 'Cloud Computing', 'Drug Discovery', 'Bioinformatics', 'Python', 'SaaS']

    return (
        <div className="min-h-screen bg-slate-50 pt-32 pb-20">
            <div className="container mx-auto px-4">

                {/* Page Header */}
                <div className="text-center mb-16">
                    <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4 tracking-tight">
                        Latest <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-secondary-500">Insights</span>
                    </h1>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Explore the latest news, tutorials, and scientific breakthroughs in computational drug discovery.
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">

                    {/* Main Content - Blog Posts */}
                    <div className="lg:col-span-2 space-y-10">
                        <div className="grid md:grid-cols-2 gap-8">
                            {posts.map(post => (
                                <article key={post.id} className="bg-white rounded-2xl shadow-sm border border-slate-200 group hover:border-primary-200 transition-all duration-300 flex flex-col h-full overflow-hidden hover:shadow-lg">
                                    {/* Image */}
                                    <div className="h-48 overflow-hidden relative">
                                        <img
                                            src={post.image}
                                            alt={post.title}
                                            className="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-500"
                                        />
                                        <div className="absolute top-4 left-4 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-bold text-primary-600 uppercase tracking-wider shadow-sm">
                                            {post.category}
                                        </div>
                                    </div>

                                    {/* Content */}
                                    <div className="p-6 flex-1 flex flex-col">
                                        <div className="flex items-center text-xs text-slate-400 mb-3 space-x-2">
                                            <span>{post.date}</span>
                                            <span>•</span>
                                            <span>{post.author}</span>
                                        </div>

                                        <h2 className="text-xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors line-clamp-2">
                                            <Link to={`/blog/${post.id}`}>{post.title}</Link>
                                        </h2>

                                        <p className="text-slate-600 text-sm mb-4 line-clamp-3 flex-1">
                                            {post.excerpt}
                                        </p>

                                        <Link
                                            to={`/blog/${post.id}`}
                                            className="inline-flex items-center text-sm font-bold text-primary-600 hover:text-primary-700 transition-colors mt-auto group/link"
                                        >
                                            Read Article
                                            <svg className="w-4 h-4 ml-1 transform group-hover/link:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                            </svg>
                                        </Link>
                                    </div>
                                </article>
                            ))}
                        </div>

                        {/* Pagination Mockup */}
                        <div className="flex justify-center space-x-2 mt-12">
                            <button className="w-10 h-10 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-slate-500 hover:bg-slate-50 hover:text-primary-600 transition">1</button>
                            <button className="w-10 h-10 rounded-lg bg-primary-600 text-white font-bold flex items-center justify-center shadow-lg shadow-primary-600/20">2</button>
                            <button className="w-10 h-10 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-slate-500 hover:bg-slate-50 hover:text-primary-600 transition">3</button>
                            <span className="w-10 h-10 flex items-center justify-center text-slate-400">...</span>
                            <button className="w-10 h-10 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-slate-500 hover:bg-slate-50 hover:text-primary-600 transition">→</button>
                        </div>
                    </div>

                    {/* Sidebar */}
                    <aside className="lg:col-span-1 space-y-8">

                        {/* Search Widget */}
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold text-slate-900 mb-4">Search</h3>
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search articles..."
                                    className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition outline-none text-sm text-slate-900 placeholder-slate-400"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                />
                                <svg className="w-5 h-5 text-slate-400 absolute left-3 top-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                            </div>
                        </div>

                        {/* Categories Widget */}
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold text-slate-900 mb-4">Categories</h3>
                            <ul className="space-y-3">
                                {categories.map((cat, idx) => (
                                    <li key={idx}>
                                        <a href="#" className="flex items-center justify-between group">
                                            <span className="text-slate-600 group-hover:text-primary-600 transition-colors text-sm font-medium">{cat.name}</span>
                                            <span className="bg-slate-100 text-slate-500 py-0.5 px-2 rounded-full text-xs font-bold group-hover:bg-primary-50 group-hover:text-primary-600 transition-colors">{cat.count}</span>
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Recent Posts Widget */}
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold text-slate-900 mb-6">Recent Posts</h3>
                            <div className="space-y-6">
                                {posts.slice(0, 3).map(post => (
                                    <div key={post.id} className="flex space-x-4 group cursor-pointer">
                                        <div className="w-16 h-16 rounded-lg overflow-hidden flex-shrink-0 border border-slate-200">
                                            <img src={post.image} alt={post.title} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300" />
                                        </div>
                                        <div>
                                            <h4 className="text-sm font-bold text-slate-900 leading-snug mb-1 group-hover:text-primary-600 transition-colors line-clamp-2">
                                                {post.title}
                                            </h4>
                                            <span className="text-xs text-slate-400">{post.date}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Tags Widget */}
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <h3 className="text-lg font-bold text-slate-900 mb-4">Popular Tags</h3>
                            <div className="flex flex-wrap gap-2">
                                {tags.map((tag, idx) => (
                                    <a
                                        key={idx}
                                        href="#"
                                        className="text-xs font-medium bg-slate-50 border border-slate-200 text-slate-600 px-3 py-1.5 rounded-lg hover:bg-primary-50 hover:text-primary-600 hover:border-primary-200 transition-all"
                                    >
                                        #{tag}
                                    </a>
                                ))}
                            </div>
                        </div>

                        {/* Newsletter Widget */}
                        <div className="bg-slate-900 rounded-2xl p-6 text-white shadow-lg shadow-slate-900/20">
                            <h3 className="text-lg font-bold mb-2">Subscribe to Newsletter</h3>
                            <p className="text-slate-400 text-sm mb-4">Get the latest updates and tutorials delivered to your inbox.</p>
                            <div className="space-y-3">
                                <input
                                    type="email"
                                    placeholder="Your email address"
                                    className="w-full px-4 py-2.5 rounded-lg bg-slate-800 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-primary-500 transition text-sm"
                                />
                                <button className="w-full py-2.5 bg-primary-600 text-white font-bold rounded-lg hover:bg-primary-500 transition shadow-sm">
                                    Subscribe
                                </button>
                            </div>
                        </div>

                    </aside>
                </div>
            </div>
        </div>
    )
}
