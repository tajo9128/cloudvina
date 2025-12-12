import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { MessageSquare, Users, TrendingUp, Search, Info, Clock, MessageCircle, Shield, TrendingDown } from 'lucide-react';
import { categories, recentTopics, stats, users } from '../data/communityData';

export default function CommunityPage() {
    const [searchQuery, setSearchQuery] = useState('');

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col">

            {/* Hero / Welcome Section */}
            <div className="bg-white border-b border-slate-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div>
                            <h1 className="text-4xl font-bold font-display text-slate-900 mb-2">
                                Community Forum
                            </h1>
                            <p className="text-lg text-slate-600">
                                Join the conversation, ask questions, and share your research.
                            </p>
                        </div>
                        <div className="flex gap-4">
                            <button className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors shadow-sm">
                                Login to Post
                            </button>
                            <button className="px-6 py-3 bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-semibold rounded-lg transition-colors">
                                Register
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Layout */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="flex flex-col lg:flex-row gap-8">

                    {/* Left Column: Forums (wpForo Layout) */}
                    <div className="lg:w-3/4 space-y-8">
                        {/* Search Bar */}
                        <div className="relative mb-6">
                            <input
                                type="text"
                                placeholder="Search the community..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-12 pr-4 py-3 bg-white border border-slate-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-lg"
                            />
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-6 h-6 text-slate-400" />
                        </div>

                        {/* Forum Categories */}
                        {categories.map(category => (
                            <div key={category.id} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                                {/* Category Header */}
                                <div className="bg-slate-50 px-6 py-3 border-b border-slate-200 flex justify-between items-center">
                                    <h2 className="text-lg font-bold text-slate-800 uppercase tracking-wide">
                                        {category.title}
                                    </h2>
                                    <div className="hidden md:flex gap-8 text-xs font-semibold text-slate-500 uppercase tracking-wider w-1/3">
                                        <span className="w-16 text-center">Topics</span>
                                        <span className="w-16 text-center">Posts</span>
                                        <span className="flex-1">Last Post</span>
                                    </div>
                                </div>

                                {/* Forums List */}
                                <div className="divide-y divide-slate-100">
                                    {category.forums.map(forum => (
                                        <div key={forum.id} className="p-6 hover:bg-slate-50 transition-colors">
                                            <div className="flex flex-col md:flex-row md:items-center gap-6">
                                                {/* Icon + Info */}
                                                <div className="flex-1 flex gap-4">
                                                    <div className="w-12 h-12 bg-primary-50 rounded-lg flex items-center justify-center text-2xl flex-shrink-0">
                                                        {forum.icon}
                                                    </div>
                                                    <div>
                                                        <Link to={`/community/forum/${forum.id}`} className="text-lg font-bold text-slate-900 hover:text-primary-600 transition-colors block mb-1">
                                                            {forum.title}
                                                        </Link>
                                                        <p className="text-slate-500 text-sm leading-relaxed">
                                                            {forum.description}
                                                        </p>
                                                    </div>
                                                </div>

                                                {/* Stats (Desktop) */}
                                                <div className="hidden md:flex items-center gap-8 w-1/3">
                                                    <div className="w-16 text-center">
                                                        <span className="block font-bold text-slate-900">{forum.topicsCount}</span>
                                                    </div>
                                                    <div className="w-16 text-center">
                                                        <span className="block font-bold text-slate-900">{forum.postsCount}</span>
                                                    </div>
                                                    <div className="flex-1 text-sm">
                                                        <div className="font-medium text-slate-900 truncate max-w-[140px]">
                                                            {forum.lastPost.title}
                                                        </div>
                                                        <div className="text-slate-500 text-xs">
                                                            by <span className="text-primary-600">{forum.lastPost.author}</span> • {forum.lastPost.time}
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Mobile Stats (Compact) */}
                                                <div className="md:hidden flex justify-between text-sm text-slate-500 border-t border-slate-100 pt-4 mt-2">
                                                    <div>
                                                        <span className="font-semibold">{forum.topicsCount}</span> Topics
                                                    </div>
                                                    <div>
                                                        <span className="font-semibold">{forum.postsCount}</span> Posts
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Right Sidebar */}
                    <aside className="lg:w-1/4 space-y-6">
                        {/* Info / Guidelines */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Info className="w-5 h-5 text-primary-600" />
                                Welcome
                            </h3>
                            <p className="text-sm text-slate-600 mb-4">
                                Please read our community guidelines before posting. Be respectful and helpful to fellow researchers.
                            </p>
                            <button className="w-full py-2 bg-slate-100 text-slate-700 font-medium rounded-lg hover:bg-slate-200 transition-colors text-sm">
                                Read Guidelines
                            </button>
                        </div>

                        {/* Recent Topics */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Clock className="w-5 h-5 text-primary-600" />
                                Recent Topics
                            </h3>
                            <div className="space-y-4">
                                {recentTopics.map(topic => (
                                    <Link key={topic.id} to={`/community/topic/${topic.id}`} className="block group cursor-pointer">
                                        <h4 className="font-medium text-slate-900 text-sm group-hover:text-primary-600 transition-colors line-clamp-2">
                                            {topic.title}
                                        </h4>
                                        <div className="flex justify-between items-center mt-1 text-xs text-slate-500">
                                            <span>by {topic.author.name}</span>
                                            <span className="flex items-center gap-1">
                                                <MessageCircle className="w-3 h-3" /> {topic.replies}
                                            </span>
                                        </div>
                                    </Link>
                                ))}
                            </div>
                        </div>

                        {/* Statistics */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-primary-600" />
                                Statistics
                            </h3>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-slate-600">Members</span>
                                    <span className="font-bold text-slate-900">{stats.members.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-600">Topics</span>
                                    <span className="font-bold text-slate-900">{stats.totalTopics.toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-600">Posts</span>
                                    <span className="font-bold text-slate-900">{stats.totalPosts.toLocaleString()}</span>
                                </div>
                                <div className="pt-3 border-t border-slate-100 flex justify-between items-center">
                                    <span className="text-slate-600 flex items-center gap-2">
                                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                        Online Now
                                    </span>
                                    <span className="font-bold text-green-600">{stats.online}</span>
                                </div>
                            </div>
                        </div>

                        {/* Staff */}
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Shield className="w-5 h-5 text-primary-600" />
                                Staff Online
                            </h3>
                            <div className="flex gap-2">
                                <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center text-white font-bold text-sm" title="Admin">A</div>
                                <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold text-sm" title="Moderator">M</div>
                            </div>
                        </div>
                    </aside>
                </div>
            </div>
        </div>
    );
}
