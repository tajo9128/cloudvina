import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { MessageSquare, Share2, Flag, ThumbsUp, MoreHorizontal, ChevronLeft, Quote, User } from 'lucide-react';
import { topics, posts, users } from '../data/communityData';

export default function TopicPage() {
    const { topicId } = useParams();
    const topicIdInt = parseInt(topicId);

    // Find topic
    const topic = topics.find(t => t.id === topicIdInt);

    // Find posts for this topic
    const topicPosts = posts.filter(p => p.topicId === topicIdInt);

    if (!topic) {
        return <div className="p-12 text-center text-slate-500">Topic not found</div>;
    }

    // Helper for user badge
    const UserBadge = ({ role }) => {
        let color = 'bg-slate-100 text-slate-600';
        if (role === 'Administrator') color = 'bg-red-100 text-red-700';
        if (role === 'Moderator') color = 'bg-purple-100 text-purple-700';
        return (
            <span className={`text-xs px-2 py-0.5 rounded font-bold uppercase tracking-wider ${color}`}>
                {role}
            </span>
        );
    };

    return (
        <div className="min-h-screen bg-slate-50">
            {/* Topic Header */}
            <div className="bg-white border-b border-slate-200 sticky top-0 z-20 shadow-sm">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                    <div className="flex items-center gap-2 mb-2 text-sm text-slate-500">
                        <Link to="/community" className="hover:text-primary-600">Community</Link>
                        <span>/</span>
                        <Link to={`/community/forum/${topic.forumId}`} className="hover:text-primary-600">Forum</Link>
                        <span>/</span>
                        <span className="text-slate-900 truncate max-w-[200px]">{topic.title}</span>
                    </div>

                    <div className="flex justify-between items-start gap-4">
                        <h1 className="text-2xl md:text-3xl font-bold font-display text-slate-900">
                            {topic.title}
                        </h1>
                        <button className="hidden md:flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 font-medium transition-colors">
                            <MessageSquare className="w-4 h-4" /> Reply
                        </button>
                    </div>

                    <div className="flex items-center gap-4 mt-4 text-sm text-slate-500">
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 bg-slate-200 rounded-full flex items-center justify-center text-xs font-bold text-slate-600">
                                {topic.author.name.charAt(0)}
                            </div>
                            <span className="font-medium text-slate-700">{topic.author.name}</span>
                        </div>
                        <span>•</span>
                        <span>{new Date(topic.createdAt).toLocaleString()}</span>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                            <ThumbsUp className="w-3 h-3" /> {topic.views} Views
                        </span>
                    </div>
                </div>
            </div>

            {/* Posts Stream */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">

                {topicPosts.map((post, index) => (
                    <div key={post.id} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden" id={`post-${post.id}`}>
                        {/* Post Header (Mobile only) */}
                        <div className="md:hidden p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-slate-200 rounded-full flex items-center justify-center text-xs font-bold">
                                    {post.author.name.charAt(0)}
                                </div>
                                <div>
                                    <div className="font-bold text-sm text-slate-900">{post.author.name}</div>
                                    <div className="text-xs text-slate-500">{post.author.role}</div>
                                </div>
                            </div>
                            <span className="text-xs text-slate-400">#{index + 1}</span>
                        </div>

                        <div className="flex flex-col md:flex-row">
                            {/* Author Sidebar (Desktop) */}
                            <div className="hidden md:flex flex-col items-center w-48 bg-slate-50 p-6 border-r border-slate-200 flex-shrink-0">
                                <div className="w-20 h-20 bg-white border-2 border-slate-200 rounded-full flex items-center justify-center text-2xl font-bold text-slate-400 mb-3 shadow-sm">
                                    {post.author.avatar ? (
                                        <img src={post.author.avatar} alt={post.author.name} className="w-full h-full object-cover rounded-full" />
                                    ) : (
                                        post.author.name.charAt(0)
                                    )}
                                </div>
                                <div className="text-center">
                                    <div className="font-bold text-slate-900 mb-1">{post.author.name}</div>
                                    <UserBadge role={post.author.role} />

                                    <div className="mt-4 space-y-1 text-xs text-slate-500">
                                        <div>Posts: {post.author.posts || 0}</div>
                                        <div>Joined: {post.author.joinDate ? new Date(post.author.joinDate).toLocaleDateString() : 'N/A'}</div>
                                    </div>
                                </div>
                            </div>

                            {/* Post Body */}
                            <div className="flex-1 p-6 md:p-8">
                                <div className="flex justify-between items-start mb-6">
                                    <div className="text-xs text-slate-400">
                                        Posted {new Date(post.createdAt).toLocaleString()}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-slate-400 hidden md:inline">#{index + 1}</span>
                                        <button className="p-1 hover:bg-slate-100 rounded text-slate-400">
                                            <Share2 className="w-4 h-4" />
                                        </button>
                                        <button className="p-1 hover:bg-slate-100 rounded text-slate-400">
                                            <Flag className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>

                                {/* Content */}
                                <div
                                    className="prose prose-slate max-w-none mb-8 text-slate-800"
                                    dangerouslySetInnerHTML={{ __html: post.content }}
                                />

                                {/* Footer / Actions */}
                                <div className="flex items-center justify-between pt-6 border-t border-slate-100">
                                    <div className="flex items-center gap-4">
                                        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-600 text-sm font-medium transition-colors">
                                            <ThumbsUp className="w-4 h-4" />
                                            <span>{post.likes}</span>
                                        </button>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <button className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-slate-600 hover:bg-slate-50 text-sm font-medium transition-colors">
                                            <Quote className="w-4 h-4" /> Quote
                                        </button>
                                        <button className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary-50 text-primary-700 hover:bg-primary-100 text-sm font-medium transition-colors">
                                            <MessageSquare className="w-4 h-4" /> Reply
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}

                {/* Quick Reply Box */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 md:p-8">
                    <h3 className="font-bold text-lg text-slate-900 mb-4">Post a Reply</h3>
                    <div className="mb-4">
                        <div className="w-full h-32 p-4 border border-slate-300 rounded-lg text-slate-500 bg-slate-50 cursor-text">
                            Write your reply here... (Rich text editor coming soon)
                        </div>
                    </div>
                    <div className="flex justify-end">
                        <button className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg transition-colors shadow-lg shadow-primary-500/30">
                            Submit Reply
                        </button>
                    </div>
                </div>

            </div>
        </div>
    );
}
