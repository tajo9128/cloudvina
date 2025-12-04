import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { Search, Ban, CheckCircle, User, Users as UsersIcon, Shield, RefreshCw } from 'lucide-react';

const Users = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');

    useEffect(() => {
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();

            const response = await fetch(`${import.meta.env.VITE_API_URL}/admin/users`, {
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setUsers(data);
            }
        } catch (error) {
            console.error('Error fetching users:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSuspendUser = async (userId) => {
        const reason = prompt("Enter reason for suspension:");
        if (!reason) return;

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const response = await fetch(`${import.meta.env.VITE_API_URL}/admin/users/${userId}/suspend?reason=${encodeURIComponent(reason)}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`
                }
            });

            if (response.ok) {
                alert('User suspended successfully');
                fetchUsers();
            } else {
                alert('Failed to suspend user');
            }
        } catch (error) {
            console.error('Error suspending user:', error);
        }
    };

    const filteredUsers = users.filter(user =>
        user.email?.toLowerCase().includes(search.toLowerCase()) ||
        user.username?.toLowerCase().includes(search.toLowerCase()) ||
        user.id?.includes(search)
    );

    return (
        <div className="p-8">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
                        <UsersIcon className="text-secondary-400" /> User Management
                    </h1>
                    <p className="text-slate-400">Manage user accounts and access</p>
                </div>
                <button
                    onClick={fetchUsers}
                    className="flex items-center gap-2 px-4 py-2 bg-secondary-600/20 text-secondary-400 border border-secondary-500/30 rounded-lg hover:bg-secondary-600/30 transition-colors"
                >
                    <RefreshCw size={16} />
                    Refresh
                </button>
            </header>

            {/* Search */}
            <div className="mb-6 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
                <input
                    type="text"
                    placeholder="Search users by email, username, or ID..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-primary-500/20 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/30 transition-all"
                />
            </div>

            {/* Users Table */}
            <div className="bg-slate-800/30 border border-primary-500/20 rounded-xl overflow-hidden backdrop-blur-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead>
                            <tr className="bg-slate-800/50 border-b border-primary-500/20 text-slate-400 text-sm">
                                <th className="p-4 font-medium">User</th>
                                <th className="p-4 font-medium">Role</th>
                                <th className="p-4 font-medium">Joined</th>
                                <th className="p-4 font-medium">Status</th>
                                <th className="p-4 text-right font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-primary-500/10">
                            {loading ? (
                                <tr>
                                    <td colSpan="5" className="p-8 text-center text-slate-400">
                                        <div className="flex items-center justify-center gap-2">
                                            <RefreshCw className="animate-spin" size={16} />
                                            Loading users...
                                        </div>
                                    </td>
                                </tr>
                            ) : filteredUsers.length === 0 ? (
                                <tr>
                                    <td colSpan="5" className="p-8 text-center text-slate-400">No users found</td>
                                </tr>
                            ) : (
                                filteredUsers.map(user => (
                                    <tr key={user.id} className="hover:bg-primary-500/5 transition-colors">
                                        <td className="p-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary-500/30 to-secondary-500/30 border border-primary-500/30 flex items-center justify-center text-primary-300">
                                                    <User size={18} />
                                                </div>
                                                <div>
                                                    <div className="text-white font-medium">{user.email}</div>
                                                    <div className="text-slate-500 text-xs font-mono">{user.id?.substring(0, 12)}...</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            {user.is_admin ? (
                                                <span className="px-3 py-1 bg-gradient-to-r from-secondary-500/20 to-secondary-600/10 text-secondary-400 rounded-full text-xs font-bold border border-secondary-500/30 flex items-center gap-1 w-fit">
                                                    <Shield size={12} /> ADMIN
                                                </span>
                                            ) : (
                                                <span className="px-3 py-1 bg-slate-700/50 text-slate-300 rounded-full text-xs border border-slate-600/50">USER</span>
                                            )}
                                        </td>
                                        <td className="p-4 text-slate-400 text-sm">
                                            {new Date(user.created_at).toLocaleDateString()}
                                        </td>
                                        <td className="p-4">
                                            <span className="flex items-center gap-1 text-green-400 text-sm bg-green-500/10 px-2 py-1 rounded-full border border-green-500/30 w-fit">
                                                <CheckCircle size={14} /> Active
                                            </span>
                                        </td>
                                        <td className="p-4 text-right">
                                            {!user.is_admin && (
                                                <button
                                                    onClick={() => handleSuspendUser(user.id)}
                                                    className="text-red-400 hover:text-red-300 text-sm font-medium flex items-center gap-1 ml-auto bg-red-500/10 px-3 py-1 rounded-lg border border-red-500/30 hover:bg-red-500/20 transition-colors"
                                                >
                                                    <Ban size={14} /> Suspend
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default Users;
