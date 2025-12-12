import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, MapPin, Calendar, MessageSquare, Star, Shield, Filter } from 'lucide-react';
import { members } from '../data/communityData';

export default function MembersPage() {
    const [searchTerm, setSearchTerm] = useState('');
    const [roleFilter, setRoleFilter] = useState('All');

    const filteredMembers = members.filter(member => {
        const matchesSearch = member.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            member.username.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesRole = roleFilter === 'All' || member.role === roleFilter;
        return matchesSearch && matchesRole;
    });

    return (
        <div className="min-h-screen bg-slate-50 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-slate-900 mb-2">Member Directory</h1>
                    <p className="text-slate-500">Connect with {members.length} researchers and students in our community.</p>
                </div>

                {/* Filters */}
                <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-8 flex flex-col md:flex-row gap-4 justify-between items-center">
                    <div className="relative w-full md:w-96">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
                        <input
                            type="text"
                            placeholder="Find a member..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
                        />
                    </div>

                    <div className="flex items-center gap-2 w-full md:w-auto overflow-x-auto pb-2 md:pb-0">
                        <Filter className="w-5 h-5 text-slate-400 hidden md:block" />
                        {['All', 'Admin', 'Moderator', 'Member'].map(role => (
                            <button
                                key={role}
                                onClick={() => setRoleFilter(role)}
                                className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${roleFilter === role ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                            >
                                {role}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredMembers.map(member => (
                        <div key={member.id} className="bg-white rounded-xl border border-slate-200 p-6 hover:shadow-lg transition-all group relative overflow-hidden">
                            {/* Role Badge */}
                            <div className={`absolute top-0 right-0 px-3 py-1 text-[10px] font-bold uppercase tracking-wider rounded-bl-xl ${member.role === 'Admin' ? 'bg-red-500 text-white' :
                                    member.role === 'Moderator' ? 'bg-green-500 text-white' :
                                        'bg-slate-100 text-slate-500'
                                }`}>
                                {member.role}
                            </div>

                            <div className="flex items-start gap-4 mb-4">
                                <img src={member.avatar} alt={member.name} className="w-16 h-16 rounded-full border-2 border-slate-100 group-hover:border-primary-500 transition-colors" />
                                <div>
                                    <h3 className="font-bold text-lg text-slate-900 leading-tight">
                                        <Link to={`/community/profile/${member.username}`} className="hover:text-primary-600 transition-colors">
                                            {member.name}
                                        </Link>
                                    </h3>
                                    <div className="text-primary-600 text-sm font-medium">@{member.username}</div>
                                    <div className="flex items-center gap-1 text-slate-400 text-xs mt-1">
                                        <MapPin className="w-3 h-3" /> {member.location}
                                    </div>
                                </div>
                            </div>

                            <p className="text-slate-600 text-sm mb-6 line-clamp-2 min-h-[40px]">
                                {member.bio}
                            </p>

                            <div className="grid grid-cols-3 gap-2 border-t border-slate-100 pt-4 text-center">
                                <div>
                                    <div className="flex items-center justify-center gap-1 text-amber-500 font-bold">
                                        <Star className="w-3 h-3 fill-current" /> {member.reputation}
                                    </div>
                                    <div className="text-[10px] text-slate-400 uppercase tracking-wider">Rep</div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-center gap-1 text-slate-700 font-bold">
                                        <MessageSquare className="w-3 h-3" /> {member.posts}
                                    </div>
                                    <div className="text-[10px] text-slate-400 uppercase tracking-wider">Posts</div>
                                </div>
                                <div>
                                    <div className="flex items-center justify-center gap-1 text-slate-700 font-bold">
                                        <Calendar className="w-3 h-3" /> {member.joinedDate}
                                    </div>
                                    <div className="text-[10px] text-slate-400 uppercase tracking-wider">Joined</div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
