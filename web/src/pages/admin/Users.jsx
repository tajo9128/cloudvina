import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
// import { Search, Ban, CheckCircle, User, Users as UsersIcon, Shield, RefreshCw, Plus, Trash2, Edit2, X } from 'lucide-react';

const Search = () => <span>🔍</span>;
const Ban = () => <span>🚫</span>;
const CheckCircle = () => <span>✅</span>;
const User = () => <span>👤</span>;
const UsersIcon = () => <span>👥</span>;
const Shield = () => <span>🛡️</span>;
const RefreshCw = () => <span>🔄</span>;
const Plus = () => <span>➕</span>;
const Trash2 = () => <span>🗑️</span>;
const Edit2 = () => <span>✏️</span>;
const X = () => <span>❌</span>;

const Users = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');

    // Modal states
    const [showAddModal, setShowAddModal] = useState(false);
    const [showEditModal, setShowEditModal] = useState(false);
    const [selectedUser, setSelectedUser] = useState(null);

    // Form states
    const [formData, setFormData] = useState({
        email: '',
        password: '',
        credits: 10,
        is_admin: false,
        role: 'user',
        plan: 'free'
    });

    useEffect(() => {
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/users`, {
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
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

    const handleAddUser = async (e) => {
        e.preventDefault();
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/users`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.access_token}`
                },
                body: JSON.stringify(formData)
            });

            if (response.ok) {
                alert('User created successfully');
                setShowAddModal(false);
                fetchUsers();
                setFormData({ email: '', password: '', credits: 10, is_admin: false, role: 'user', plan: 'free' });
            } else {
                const err = await response.json();
                alert('Failed to create user: ' + (err.detail || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error creating user:', error);
        }
    };

    const handleEditUser = async (e) => {
        e.preventDefault();
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/users/${selectedUser.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.access_token}`
                },
                body: JSON.stringify({
                    credits: parseInt(formData.credits),
                    is_admin: formData.is_admin,
                    role: formData.role,
                    plan: formData.plan
                })
            });

            if (response.ok) {
                alert('User updated successfully');
                setShowEditModal(false);
                fetchUsers();
            } else {
                alert('Failed to update user');
            }
        } catch (error) {
            console.error('Error updating user:', error);
        }
    };

    const handleDeleteUser = async (userId) => {
        if (!confirm('Are you sure you want to PERMANENTLY delete this user? This cannot be undone.')) return;

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/users/${userId}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });

            if (response.ok) {
                alert('User deleted successfully');
                fetchUsers();
            } else {
                const err = await response.json();
                alert('Failed to delete user: ' + (err.detail || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error deleting user:', error);
        }
    };

    const handlePaymentBlockQuery = async (user) => {
        const isSuspended = user.role === 'suspended';
        const action = isSuspended ? 'Resume Access' : 'Block Access';
        const confirmMsg = isSuspended
            ? `Resume access for ${user.email}?\n(Payment Received)`
            : `Block access for ${user.email}?\n(Mark as Payment Overdue)`;

        if (!confirm(confirmMsg)) return;

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const newRole = isSuspended ? 'user' : 'suspended';

            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/users/${user.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${session?.access_token}`
                },
                body: JSON.stringify({ role: newRole })
            });

            if (response.ok) {
                alert(`User ${isSuspended ? 'resumed' : 'blocked'} successfully.`);
                fetchUsers();
            } else {
                alert('Failed to update status');
            }
        } catch (error) {
            console.error(error);
        }
    };

    const openEditModal = (user) => {
        setSelectedUser(user);
        setFormData({
            ...formData,
            credits: user.credits || 0,
            is_admin: user.is_admin || false,
            role: user.role || 'user',
            plan: user.plan || 'free'
        });
        setShowEditModal(true);
    };

    const filteredUsers = users.filter(user =>
        user.email?.toLowerCase().includes(search.toLowerCase()) ||
        user.username?.toLowerCase().includes(search.toLowerCase()) ||
        user.id?.includes(search)
    );

    // Calculate Stats
    const stats = {
        free: users.filter(u => !u.plan || u.plan === 'free').length,
        pro: users.filter(u => u.plan === 'pro').length,
        premium: users.filter(u => u.plan === 'premium').length
    };

    return (
        <div className="p-8">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
                        <UsersIcon className="text-secondary-400" /> User Management
                    </h1>
                    <p className="text-slate-400">Manage user accounts, roles, and credits</p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={fetchUsers}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-slate-300 border border-slate-700 rounded-lg hover:bg-slate-700 transition-colors"
                    >
                        <RefreshCw size={16} /> Refresh
                    </button>
                    <button
                        onClick={() => { setShowAddModal(true); setFormData({ email: '', password: '', credits: 10, is_admin: false, role: 'user', plan: 'free' }); }}
                        className="flex items-center gap-2 px-4 py-2 bg-secondary-600 text-white rounded-lg hover:bg-secondary-500 shadow-lg shadow-secondary-600/20 transition-all transform hover:-translate-y-0.5"
                    >
                        <Plus size={18} /> Add User
                    </button>
                </div>
            </header>

            {/* Plan Distribution Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="text-slate-400 text-sm font-medium mb-1">Free Users</p>
                            <h3 className="text-3xl font-bold text-white">{stats.free}</h3>
                        </div>
                        <div className="p-2 bg-slate-700 rounded-lg text-slate-300">
                            <User size={20} />
                        </div>
                    </div>
                    <div className="mt-2 text-xs text-slate-500">Standard Access (10 credits/mo)</div>
                </div>
                <div className="bg-slate-800 p-6 rounded-xl border border-indigo-500/30">
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="text-indigo-400 text-sm font-medium mb-1">Pro Users</p>
                            <h3 className="text-3xl font-bold text-white">{stats.pro}</h3>
                        </div>
                        <div className="p-2 bg-indigo-500/20 rounded-lg text-indigo-400">
                            <Shield size={20} />
                        </div>
                    </div>
                    <div className="mt-2 text-xs text-indigo-300/60">Research Bundle (50 credits)</div>
                </div>
                <div className="bg-slate-800 p-6 rounded-xl border border-purple-500/30">
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="text-purple-400 text-sm font-medium mb-1">Premium Users</p>
                            <h3 className="text-3xl font-bold text-white">{stats.premium}</h3>
                        </div>
                        <div className="p-2 bg-purple-500/20 rounded-lg text-purple-400">
                            <CheckCircle size={20} />
                        </div>
                    </div>
                    <div className="mt-2 text-xs text-purple-300/60">Enterprise Bundle (100+ credits)</div>
                </div>
            </div>

            {/* Search */}
            <div className="mb-6 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
                <input
                    type="text"
                    placeholder="Search users..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-primary-500/20 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500/50 transition-all font-mono text-sm"
                />
            </div>

            {/* Users Table */}
            <div className="bg-slate-800/30 border border-primary-500/20 rounded-xl overflow-hidden backdrop-blur-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead>
                            <tr className="bg-slate-800/50 border-b border-primary-500/20 text-slate-400 text-sm">
                                <th className="p-4 font-medium">User</th>
                                <th className="p-4 font-medium">Plan</th>
                                <th className="p-4 font-medium">Role</th>
                                <th className="p-4 font-medium">Credits</th>
                                <th className="p-4 font-medium">Status</th>
                                <th className="p-4 text-right font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-primary-500/10">
                            {loading ? (
                                <tr><td colSpan="6" className="p-8 text-center text-slate-400">Loading...</td></tr>
                            ) : filteredUsers.length === 0 ? (
                                <tr><td colSpan="6" className="p-8 text-center text-slate-400">No users found</td></tr>
                            ) : (
                                filteredUsers.map(user => (
                                    <tr key={user.id} className="hover:bg-primary-500/5 transition-colors">
                                        <td className="p-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-full bg-slate-700/50 flex items-center justify-center text-slate-300 font-bold">
                                                    {user.email?.charAt(0).toUpperCase()}
                                                </div>
                                                <div>
                                                    <div className="text-white font-medium">{user.email}</div>
                                                    <div className="text-slate-500 text-xs font-mono">{user.id?.substring(0, 8)}...</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <span className={`px-2 py-1 rounded text-xs uppercase font-bold tracking-wider ${user.plan === 'premium' ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' :
                                                user.plan === 'pro' ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30' :
                                                    'bg-slate-700 text-slate-400'
                                                }`}>
                                                {user.plan || 'FREE'}
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <div className="flex gap-2">
                                                {user.is_admin && (
                                                    <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-xs border border-red-500/30 flex items-center gap-1">
                                                        <Shield size={10} /> ADMIN
                                                    </span>
                                                )}
                                                <span className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs capitalize">
                                                    {user.role || 'user'}
                                                </span>
                                            </div>
                                        </td>
                                        <td className="p-4 font-mono text-primary-400 font-bold">
                                            {user.credits}
                                        </td>
                                        <td className="p-4">
                                            <span className="flex items-center gap-1 text-green-400 text-sm">
                                                <CheckCircle size={14} /> Active
                                            </span>
                                        </td>
                                        <td className="p-4 text-right">
                                            <div className="flex justify-end gap-2">
                                                {/* Payment Block Toggle */}
                                                <button
                                                    onClick={() => handlePaymentBlockQuery(user)}
                                                    className={`p-2 rounded-lg transition-colors ${user.role === 'suspended'
                                                        ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                                        : 'bg-slate-700/50 text-slate-400 hover:bg-yellow-500/20 hover:text-yellow-400'
                                                        }`}
                                                    title={user.role === 'suspended' ? "Resume: Payment Received" : "Block: Payment Overdue"}
                                                >
                                                    {user.role === 'suspended' ? <CheckCircle size={16} /> : <Ban size={16} />}
                                                </button>

                                                <button
                                                    onClick={() => openEditModal(user)}
                                                    className="p-2 bg-slate-700/50 hover:bg-primary-500/20 text-slate-400 hover:text-primary-400 rounded-lg transition-colors"
                                                    title="Edit User"
                                                >
                                                    <Edit2 size={16} />
                                                </button>
                                                {!user.is_admin && (
                                                    <button
                                                        onClick={() => handleDeleteUser(user.id)}
                                                        className="p-2 bg-slate-700/50 hover:bg-red-500/20 text-slate-400 hover:text-red-400 rounded-lg transition-colors"
                                                        title="Delete User"
                                                    >
                                                        <Trash2 size={16} />
                                                    </button>
                                                )}
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Add User Modal */}
            {
                showAddModal && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
                        <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-md p-6 shadow-2xl">
                            <div className="flex justify-between items-center mb-6">
                                <h3 className="text-xl font-bold text-white">Add New User</h3>
                                <button onClick={() => setShowAddModal(false)} className="text-slate-400 hover:text-white"><X size={24} /></button>
                            </div>
                            <form onSubmit={handleAddUser} className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-slate-400 mb-1">Email</label>
                                    <input type="email" required className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-secondary-500 focus:outline-none"
                                        value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-slate-400 mb-1">Password</label>
                                    <input type="password" required minLength={6} className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-secondary-500 focus:outline-none"
                                        value={formData.password} onChange={e => setFormData({ ...formData, password: e.target.value })} />
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-slate-400 mb-1">Initial Credits</label>
                                        <input type="number" required className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-secondary-500 focus:outline-none"
                                            value={formData.credits} onChange={e => setFormData({ ...formData, credits: parseInt(e.target.value) })} />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-400 mb-1">Plan</label>
                                        <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-secondary-500 focus:outline-none"
                                            value={formData.plan} onChange={e => setFormData({ ...formData, plan: e.target.value })}>
                                            <option value="free">Free</option>
                                            <option value="pro">Pro</option>
                                            <option value="premium">Premium</option>
                                        </select>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 pt-2">
                                    <input type="checkbox" id="isAdmin" className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-secondary-500 focus:ring-secondary-500"
                                        checked={formData.is_admin} onChange={e => setFormData({ ...formData, is_admin: e.target.checked })} />
                                    <label htmlFor="isAdmin" className="text-sm text-slate-300">Grant Admin Privileges</label>
                                </div>
                                <button type="submit" className="w-full bg-secondary-600 hover:bg-secondary-500 text-white font-bold py-3 rounded-xl mt-4 transition-all">Create User</button>
                            </form>
                        </div>
                    </div>
                )
            }

            {/* Edit User Modal */}
            {
                showEditModal && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
                        <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-md p-6 shadow-2xl">
                            <div className="flex justify-between items-center mb-6">
                                <h3 className="text-xl font-bold text-white">Edit User: {selectedUser?.email}</h3>
                                <button onClick={() => setShowEditModal(false)} className="text-slate-400 hover:text-white"><X size={24} /></button>
                            </div>
                            <form onSubmit={handleEditUser} className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-slate-400 mb-1">Credits</label>
                                        <input type="number" required className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-secondary-500 focus:outline-none"
                                            value={formData.credits} onChange={e => setFormData({ ...formData, credits: e.target.value })} />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-slate-400 mb-1">Plan</label>
                                        <select className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:border-secondary-500 focus:outline-none"
                                            value={formData.plan} onChange={e => setFormData({ ...formData, plan: e.target.value })}>
                                            <option value="free">Free</option>
                                            <option value="pro">Pro (₹100)</option>
                                            <option value="premium">Premium (₹500)</option>
                                        </select>
                                    </div>
                                </div>
                                <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                                    <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">Danger Zone</h4>
                                    <div className="flex items-center gap-2">
                                        <input type="checkbox" id="editIsAdmin" className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-red-500 focus:ring-red-500"
                                            checked={formData.is_admin} onChange={e => setFormData({ ...formData, is_admin: e.target.checked })} />
                                        <label htmlFor="editIsAdmin" className="text-sm text-slate-300">Is Admin</label>
                                    </div>
                                </div>
                                <button type="submit" className="w-full bg-primary-600 hover:bg-primary-500 text-white font-bold py-3 rounded-xl mt-4 transition-all">Save Changes</button>
                            </form>
                        </div>
                    </div>
                )
            }
        </div >
    );
};

export default Users;
