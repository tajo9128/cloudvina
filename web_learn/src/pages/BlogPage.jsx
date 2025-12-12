import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Calendar, User, Clock, ArrowRight, Search, Tag, ChevronLeft, ChevronRight } from 'lucide-react';
import { blogPosts, categories, searchPosts } from '../data/blogData';

// Pagination constants
const POSTS_PER_PAGE = 6;

export default function BlogPage() {
    const [selectedCategory, setSelectedCategory] = useState('All');
    const [searchQuery, setSearchQuery] = useState('');
    const [currentPage, setCurrentPage] = useState(1);

    // Filter posts based on category and search
    const filteredPosts = blogPosts.filter(post => {
        const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory;
        const matchesSearch = post.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            post.excerpt.toLowerCase().includes(searchQuery.toLowerCase()) ||
            post.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
        return matchesCategory && matchesSearch;
    });

    // Pagination logic
    const totalPages = Math.ceil(filteredPosts.length / POSTS_PER_PAGE);
    const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
    const currentPosts = filteredPosts.slice(startIndex, startIndex + POSTS_PER_PAGE);

    // Reset pagination when filter changes
    React.useEffect(() => {
        setCurrentPage(1);
    }, [selectedCategory, searchQuery]);

    const handlePageChange = (page) => {
        if (page >= 1 && page <= totalPages) {
            setCurrentPage(page);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    };

    // Derived popular tags (could be from data)
    const popularTags = ['Tutorial', 'AutoDock', 'Virtual Screening', 'PDB Files', 'API'];

    return (
        <div className="min-h-screen bg-white flex flex-col">
            {/* Blog Header */}
            <div className="bg-slate-50 border-b border-slate-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                    <h1 className="text-4xl font-bold font-display text-slate-900 mb-4">
                        BioDockify Blog
                    </h1>
                    <p className="text-lg text-slate-600 max-w-2xl">
                        Learn about molecular docking, drug discovery, and get the latest updates from our team
                    </p>
                </div>
            </div>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="flex flex-col lg:flex-row gap-12">
                    {/* Blog Posts - Left Column */}
                    <div className="lg:w-2/3">
                        {/* Category Filter */}
                        <div className="flex items-center gap-3 mb-8 overflow-x-auto pb-2 scrollbar-hide">
                            <button
                                onClick={() => setSelectedCategory('All')}
                                className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${selectedCategory === 'All'
                                    ? 'bg-primary-600 text-white'
                                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                                    }`}
                            >
                                All
                            </button>
                            {Object.keys(categories).map(category => (
                                <button
                                    key={category}
                                    onClick={() => setSelectedCategory(category)}
                                    className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${selectedCategory === category
                                        ? 'bg-primary-600 text-white'
                                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                                        }`}
                                >
                                    {category}
                                </button>
                            ))}
                        </div>

                        {/* Blog Post Grid */}
                        <div className="space-y-8">
                            {currentPosts.length > 0 ? (
                                currentPosts.map(post => (
                                    <article key={post.id} className="group bg-white border border-slate-200 rounded-xl overflow-hidden hover:shadow-lg transition-all">
                                        <div className="md:flex">
                                            {/* Post Image */}
                                            <div className="md:w-2/5 h-64 md:h-auto overflow-hidden">
                                                <img
                                                    src={post.image}
                                                    alt={post.title}
                                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                                />
                                            </div>

                                            {/* Post Content */}
                                            <div className="md:w-3/5 p-6 flex flex-col justify-between">
                                                <div>
                                                    <div className="flex items-center gap-2 text-sm text-primary-600 font-semibold mb-3">
                                                        <Tag className="w-4 h-4" />
                                                        {post.category}
                                                    </div>

                                                    <Link to={`/blog/${post.slug}`}>
                                                        <h2 className="text-2xl font-bold font-display text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">
                                                            {post.title}
                                                        </h2>
                                                    </Link>

                                                    <p className="text-slate-600 mb-4 line-clamp-2">
                                                        {post.excerpt}
                                                    </p>
                                                </div>

                                                <div>
                                                    <div className="flex items-center gap-4 text-sm text-slate-500 mb-4">
                                                        <div className="flex items-center gap-1.5">
                                                            <User className="w-4 h-4" />
                                                            {post.author}
                                                        </div>
                                                        <div className="flex items-center gap-1.5">
                                                            <Calendar className="w-4 h-4" />
                                                            {new Date(post.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                                                        </div>
                                                        <div className="flex items-center gap-1.5">
                                                            <Clock className="w-4 h-4" />
                                                            {post.readTime}
                                                        </div>
                                                    </div>

                                                    <Link
                                                        to={`/blog/${post.slug}`}
                                                        className="inline-flex items-center gap-2 text-primary-600 hover:text-primary-700 font-semibold"
                                                    >
                                                        Read More
                                                        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                                                    </Link>
                                                </div>
                                            </div>
                                        </div>
                                    </article>
                                ))
                            ) : (
                                <div className="text-center py-12 text-slate-500">
                                    No posts found matching your criteria.
                                </div>
                            )}
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div className="flex justify-center items-center gap-2 mt-12">
                                <button
                                    onClick={() => handlePageChange(currentPage - 1)}
                                    disabled={currentPage === 1}
                                    className={`px-4 py-2 border rounded-lg flex items-center gap-1 transition-colors ${currentPage === 1
                                        ? 'border-slate-200 text-slate-400 cursor-not-allowed'
                                        : 'border-slate-300 text-slate-700 hover:bg-slate-50'
                                        }`}
                                >
                                    <ChevronLeft className="w-4 h-4" /> Previous
                                </button>

                                {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                                    <button
                                        key={page}
                                        onClick={() => handlePageChange(page)}
                                        className={`w-10 h-10 rounded-lg font-medium transition-colors ${currentPage === page
                                            ? 'bg-primary-600 text-white'
                                            : 'border border-slate-300 text-slate-700 hover:bg-slate-50'
                                            }`}
                                    >
                                        {page}
                                    </button>
                                ))}

                                <button
                                    onClick={() => handlePageChange(currentPage + 1)}
                                    disabled={currentPage === totalPages}
                                    className={`px-4 py-2 border rounded-lg flex items-center gap-1 transition-colors ${currentPage === totalPages
                                        ? 'border-slate-200 text-slate-400 cursor-not-allowed'
                                        : 'border-slate-300 text-slate-700 hover:bg-slate-50'
                                        }`}
                                >
                                    Next <ChevronRight className="w-4 h-4" />
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Sidebar - Right Column */}
                    <aside className="lg:w-1/3">
                        {/* Search */}
                        <div className="bg-slate-50 rounded-xl p-6 mb-8 sticky top-24">
                            <h3 className="font-bold text-slate-900 mb-4">Search</h3>
                            <div className="relative mb-6">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder="Search articles..."
                                    className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
                                />
                            </div>

                            {/* Popular Tags */}
                            <h3 className="font-bold text-slate-900 mb-4">Popular Tags</h3>
                            <div className="flex flex-wrap gap-2 mb-8">
                                {popularTags.map(tag => (
                                    <button
                                        key={tag}
                                        onClick={() => setSearchQuery(tag)}
                                        className="px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-sm text-slate-700 hover:border-primary-500 hover:text-primary-600 transition-colors"
                                    >
                                        {tag}
                                    </button>
                                ))}
                            </div>

                            {/* Recent Posts - Updated to show actual recent ones */}
                            <h3 className="font-bold text-slate-900 mb-4">Recent Posts</h3>
                            <div className="space-y-4">
                                {blogPosts.slice(0, 4).map(post => (
                                    <Link key={post.id} to={`/blog/${post.slug}`} className="block group">
                                        <div className="flex gap-3">
                                            <img
                                                src={post.image}
                                                alt={post.title}
                                                className="w-16 h-16 rounded-lg object-cover flex-shrink-0"
                                            />
                                            <div>
                                                <h4 className="font-semibold text-sm text-slate-900 group-hover:text-primary-600 transition-colors line-clamp-2">
                                                    {post.title}
                                                </h4>
                                                <p className="text-xs text-slate-500 mt-1">
                                                    {new Date(post.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                                                </p>
                                            </div>
                                        </div>
                                    </Link>
                                ))}
                            </div>
                        </div>
                    </aside>
                </div>
            </div>
        </div>
    );
}
