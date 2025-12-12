import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { MessageSquare, Users, Pin, Lock, Search, Filter, Plus, ChevronLeft, Calendar, Eye, MessageCircle } from 'lucide-react';
import { categories, topics, users } from '../data/communityData';

export default function ForumPage() {
    const { forumId } = useParams();
    const forumIdInt = parseInt(forumId);

    // Find the forum from the categories structure
    let forum = null;
    for (const cat of categories) {
        const found = cat.forums.find(f => f.id === forumIdInt);
        if (found) {
            forum = found;
            break;
        }
    }

    // Filter topics for this forum
    const forumTopics = topics.filter(t => t.forumId === forumIdInt);

    // Sort logic (mock)
    const [sortBy, setSortBy] = useState('recent'); // recent, views, replies

    if (!forum) {
        return <div className="p-12 text-center text-slate-500">Forum not found</div>;
    }

    return (
        <div className="min-h-screen bg-slate-50">
            {/* Forum Header */}
            <div className="bg-white border-b border-slate-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                    <div className="mb-4">
                        <Link to="/community" className="text-slate-500 hover:text-primary-600 flex items-center gap-1 text-sm">
                            <ChevronLeft className="w-4 h-4" /> Back to Community
                        </Link>
                    </div>

                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div className="flex items-start gap-4">
                            <div className="w-16 h-16 bg-primary-50 rounded-xl flex items-center justify-center text-3xl flex-shrink-0 border border-primary-100">
                                {forum.icon}
                            </div>
                            <div>
                                <h1 className="text-3xl font-bold font-display text-slate-900 mb-2">
                                    {forum.title}
                                </h1>
                                <p className="text-slate-600 max-w-2xl">
                                    {forum.description}
                                </p>
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <button className="px-4 py-2 bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium rounded-lg transition-colors flex items-center gap-2">
                                <Search className="w-4 h-4" /> Search
                            </button>
                            <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors shadow-sm flex items-center gap-2">
                                <Plus className="w-4 h-4" /> New Topic
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Topics List Container */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

                {/* Controls / Filter Bar */}
                <div className="bg-white rounded-t-xl border border-slate-200 px-6 py-4 flex justify-between items-center mb-[-1px] z-10 relative">
                    <div className="flex items-center gap-4">
                        <span className="text-sm font-semibold text-slate-700">Sort by:</span>
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value)}
                            className="bg-slate-50 border border-slate-300 rounded-md text-sm py-1 pl-2 pr-8 focus:outline-none focus:ring-1 focus:ring-primary-500"
                        >
                            <option value="recent">Recent</option>
                            <option value="views">Most Viewed</option>
                            <option value="replies">Most Replied</option>
                        </select>
                    </div>
                    <div className="text-sm text-slate-500">
                        Showing {forumTopics.length} topics
                    </div>
                </div>

                {/* Topics List (wpForo Extended Layout) */}
                <div className="bg-white rounded-b-xl border border-slate-200 overflow-hidden shadow-sm">
                    {forumTopics.length > 0 ? (
                        <div className="divide-y divide-slate-100">
                            {forumTopics.map(topic => (
                                <div key={topic.id} className="p-6 hover:bg-slate-50 transition-colors group">
                                    <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">

                                        {/* Avatar & Status Icon */}
                                        <div className="flex-shrink-0 relative">
                                            <div className="w-12 h-12 bg-slate-200 rounded-full flex items-center justify-center text-slate-500 font-bold text-lg overflow-hidden">
                                                {topic.author.avatar ? (
                                                    <img src={topic.author.avatar} alt={topic.author.name} className="w-full h-full object-cover" />
                                                ) : (
                                                    topic.author.name.charAt(0)
                                                )}
                                            </div>
                                            {topic.isSticky && (
                                                <div className="absolute -top-1 -right-1 bg-amber-500 text-white p-0.5 rounded-full border-2 border-white" title="Pinned">
                                                    <Pin className="w-3 h-3" />
                                                </div>
                                            )}
                                        </div>

                                        {/* Main Content */}
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2 mb-1">
                                                {topic.isSticky && (
                                                    <span className="px-2 py-0.5 bg-amber-100 text-amber-700 text-xs font-bold rounded uppercase tracking-wider">
                                                        Sticky
                                                    </span>
                                                )}
                                                {topic.isSolved && (
                                                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-bold rounded uppercase tracking-wider">
                                                        Solved
                                                    </span>
                                                )}
                                                <Link to={`/community/topic/${topic.id}`} className="text-lg font-bold text-slate-900 group-hover:text-primary-600 transition-colors truncate block">
                                                    {topic.title}
                                                </Link>
                                            </div>

                                            <div className="flex items-center gap-4 text-xs text-slate-500">
                                                <span className="flex items-center gap-1">
                                                    By <span className="font-medium text-slate-700">{topic.author.name}</span>
                                                </span>
                                                <span className="flex items-center gap-1">
                                                    <Calendar className="w-3 h-3" />
                                                    {new Date(topic.createdAt).toLocaleDateString()}
                                                </span>
                                                <div className="flex gap-2">
                                                    {topic.tags.map(tag => (
                                                        <span key={tag} className="bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded hover:bg-slate-200 cursor-pointer">
                                                            #{tag}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Stats */}
                                        <div className="hidden md:flex items-center gap-6 px-4 border-l border-slate-100">
                                            <div className="text-center w-16">
                                                <div className="text-lg font-bold text-slate-700">{topic.replies}</div>
                                                <div className="text-xs text-slate-400 uppercase tracking-wide">Replies</div>
                                            </div>
                                            <div className="text-center w-16">
                                                <div className="text-lg font-bold text-slate-700">{topic.views}</div>
                                                <div className="text-xs text-slate-400 uppercase tracking-wide">Views</div>
                                            </div>
                                        </div>

                                        {/* Last Post Info */}
                                        <div className="hidden lg:block w-48 text-right pl-4 border-l border-slate-100">
                                            <div className="text-xs text-slate-500 mb-1">Last post by</div>
                                            <div className="font-medium text-sm text-slate-900 truncate">
                                                {topic.lastPost.author.name}
                                            </div>
                                            <div className="text-xs text-slate-400">
                                                {topic.lastPost.time}
                                            </div>
                                        </div>

                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="p-12 text-center">
                            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-300">
                                <MessageSquare className="w-8 h-8" />
                            </div>
                            <h3 className="text-lg font-medium text-slate-900 mb-2">No topics yet</h3>
                            <p className="text-slate-500 mb-6">Be the first to start a conversation in this forum!</p>
                            <button className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors shadow-sm">
                                Create Topic
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
