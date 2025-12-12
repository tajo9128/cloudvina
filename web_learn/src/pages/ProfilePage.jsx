import React from 'react';
import { useParams } from 'react-router-dom';
import { MapPin, Calendar, MessageSquare, Star, Award, Grid, Clock, Hash } from 'lucide-react';
import { members, posts, topics } from '../data/communityData';

export default function ProfilePage() {
    const { username } = useParams();

    // Find member by username (case insensitive)
    // If no username param (e.g. /profile route), default to a logged-in user mockup or redirect
    const member = members.find(m => m.username.toLowerCase() === (username || 'drsmith').toLowerCase());

    if (!member) {
        return <div className="p-12 text-center text-slate-500">User not found</div>;
    }

    // Mock recent activity based on username
    const memberPosts = posts.slice(0, 3).map(post => ({
        ...post,
        topicTitle: topics.find(t => t.id === post.topicId)?.title || "Unknown Topic"
    }));

    // Mock courses (could come from course store)
    const completedCourses = [
        { title: 'Molecular Docking Fundamentals', score: '98%' },
        { title: 'Advanced Vina Scripting', score: '100%' }
    ];

    return (
        <div className="min-h-screen bg-slate-50 py-8">
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">

                {/* Profile Header */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-6">
                    <div className="h-32 bg-slate-900 relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-primary-900 to-slate-900 opacity-90"></div>
                        <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'radial-gradient(circle at 2px 2px, rgba(255,255,255,0.15) 1px, transparent 0)', backgroundSize: '24px 24px' }}></div>
                    </div>
                    <div className="px-8 pb-8">
                        <div className="relative flex justify-between items-end -mt-12 mb-6">
                            <div className="flex items-end gap-6">
                                <img
                                    src={member.avatar}
                                    alt={member.name}
                                    className="w-32 h-32 rounded-full border-4 border-white shadow-md bg-white"
                                />
                                <div className="pb-1">
                                    <h1 className="text-3xl font-bold text-slate-900">{member.name}</h1>
                                    <div className="text-slate-500 font-medium">@{member.username}</div>
                                </div>
                            </div>
                            <div className="flex gap-3">
                                <button className="px-4 py-2 bg-primary-600 text-white font-bold rounded-lg shadow-sm hover:bg-primary-700 transition-colors">
                                    Follow
                                </button>
                                <button className="px-4 py-2 border border-slate-300 text-slate-700 font-bold rounded-lg hover:bg-slate-50 transition-colors">
                                    Message
                                </button>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            <div className="md:col-span-2 space-y-6">
                                <div>
                                    <h3 className="font-bold text-slate-900 mb-2">About</h3>
                                    <p className="text-slate-600 leading-relaxed">
                                        {member.bio || "No bio available."}
                                    </p>
                                </div>

                                <div className="flex flex-wrap gap-4 text-sm text-slate-500">
                                    <div className="flex items-center gap-1">
                                        <MapPin className="w-4 h-4" /> {member.location || 'Unknown Location'}
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <Calendar className="w-4 h-4" /> Joined {member.joinedDate}
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <Hash className="w-4 h-4" /> ID: {member.id}
                                    </div>
                                </div>

                                {/* Badges */}
                                <div>
                                    <h3 className="font-bold text-slate-900 mb-3 flex items-center gap-2">
                                        <Award className="w-5 h-5 text-amber-500" /> Badges & Achievements
                                    </h3>
                                    <div className="flex gap-3">
                                        {member.badges && member.badges.length > 0 ? (
                                            member.badges.map(badge => (
                                                <span key={badge} className="px-3 py-1 bg-amber-50 text-amber-700 border border-amber-200 rounded-full text-xs font-bold uppercase tracking-wider">
                                                    {badge}
                                                </span>
                                            ))
                                        ) : (
                                            <span className="text-slate-400 italic text-sm">No badges yet</span>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Sidebar Stats */}
                            <div className="space-y-6">
                                <div className="bg-slate-50 rounded-xl p-6 border border-slate-100">
                                    <h3 className="font-bold text-slate-900 mb-4 px-1">Community Stats</h3>
                                    <div className="space-y-4">
                                        <div className="flex justify-between items-center px-1">
                                            <div className="flex items-center gap-2 text-slate-600">
                                                <Star className="w-4 h-4 text-amber-500" /> Reputation
                                            </div>
                                            <span className="font-bold text-slate-900">{member.reputation}</span>
                                        </div>
                                        <div className="flex justify-between items-center px-1">
                                            <div className="flex items-center gap-2 text-slate-600">
                                                <MessageSquare className="w-4 h-4 text-blue-500" /> Posts
                                            </div>
                                            <span className="font-bold text-slate-900">{member.posts}</span>
                                        </div>
                                        <div className="flex justify-between items-center px-1">
                                            <div className="flex items-center gap-2 text-slate-600">
                                                <Grid className="w-4 h-4 text-purple-500" /> Solutions
                                            </div>
                                            <span className="font-bold text-slate-900">12</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Recent Activity */}
                    <div className="md:col-span-2 space-y-4">
                        <h2 className="text-xl font-bold text-slate-900">Recent Activity</h2>
                        {memberPosts.map(post => (
                            <div key={post.id} className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                                <div className="flex items-start gap-3 mb-2">
                                    <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                                        <MessageSquare className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <div className="text-sm text-slate-500">
                                            Posted in <span className="font-bold text-slate-900">{post.topicTitle}</span>
                                        </div>
                                        <div className="text-xs text-slate-400">{post.date}</div>
                                    </div>
                                </div>
                                <p className="text-slate-700 line-clamp-2 pl-11">
                                    "{post.content}"
                                </p>
                            </div>
                        ))}
                    </div>

                    {/* Certifications / Courses */}
                    <div className="space-y-4">
                        <h2 className="text-xl font-bold text-slate-900">Completed Courses</h2>
                        {completedCourses.map((course, i) => (
                            <div key={i} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center text-green-600">
                                        <Award className="w-5 h-5" />
                                    </div>
                                    <div>
                                        <div className="font-bold text-slate-900 text-sm">{course.title}</div>
                                        <div className="text-xs text-slate-500">Score: {course.score}</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
}
