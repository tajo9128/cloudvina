import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
import { API_URL } from '../../config';
import { ShieldAlert, Users, Check, X, Plus } from 'lucide-react';

export default function RBACManagerPage() {
    const [users, setUsers] = useState([]);
    const [roles, setRoles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedUser, setSelectedUser] = useState(null); // For role editing modal

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) return;

            const headers = { 'Authorization': `Bearer ${session.access_token}` };

            // 1. Fetch Users
            const userRes = await fetch(`${API_URL}/admin/users?limit=50`, { headers });
            const userData = await userRes.json();

            // 2. Fetch All Roles
            const roleRes = await fetch(`${API_URL}/admin/rbac/roles`, { headers });
            const roleData = await roleRes.json();

            setUsers(userData);
            setRoles(roleData);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const fetchUserRoles = async (userId) => {
        const { data: { session } } = await supabase.auth.getSession();
        const headers = { 'Authorization': `Bearer ${session.access_token}` };
        const res = await fetch(`${API_URL}/admin/rbac/users/${userId}/roles`, { headers });
        return await res.json();
    };

    const handleAssign = async (userId, roleCode) => {
        const { data: { session } } = await supabase.auth.getSession();
        await fetch(`${API_URL}/admin/rbac/assign`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${session.access_token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: userId, role_code: roleCode })
        });
        // Refresh
        openUserModal(selectedUser);
    };

    const handleRemove = async (userId, roleCode) => {
        const { data: { session } } = await supabase.auth.getSession();
        await fetch(`${API_URL}/admin/rbac/remove`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${session.access_token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: userId, role_code: roleCode })
        });
        // Refresh
        openUserModal(selectedUser);
    };

    const openUserModal = async (user) => {
        const userRoles = await fetchUserRoles(user.id);
        setSelectedUser({ ...user, assigned_roles: userRoles });
    };

    return (
        <div className="min-h-screen bg-slate-50 p-6">
            <header className="mb-8">
                <div className="flex items-center gap-3 mb-2">
                    <ShieldAlert className="w-8 h-8 text-indigo-600" />
                    <h1 className="text-2xl font-bold text-slate-900">Role-Based Access Control</h1>
                </div>
                <p className="text-slate-600">Assign granular permissions to specialized staff members.</p>
            </header>

            {loading ? (
                <div>Loading...</div>
            ) : (
                <div className="bg-white rounded-xl shadow-sm border border-slate-200">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-slate-50 border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4 font-bold text-slate-500">User</th>
                                <th className="px-6 py-4 font-bold text-slate-500">Email</th>
                                <th className="px-6 py-4 font-bold text-slate-500">Designation</th>
                                <th className="px-6 py-4 font-bold text-slate-500">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {users.map(user => (
                                <tr key={user.id} className="hover:bg-slate-50">
                                    <td className="px-6 py-4 font-medium text-slate-900">{user.username || 'N/A'}</td>
                                    <td className="px-6 py-4 text-slate-600">{user.email}</td>
                                    <td className="px-6 py-4 text-slate-600">{user.designation}</td>
                                    <td className="px-6 py-4">
                                        <button
                                            onClick={() => openUserModal(user)}
                                            className="px-3 py-1 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 font-medium text-xs"
                                        >
                                            Manage Roles
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* MODAL */}
            {selectedUser && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
                    <div className="bg-white rounded-2xl shadow-xl w-full max-w-lg overflow-hidden">
                        <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                            <h3 className="text-lg font-bold text-slate-900">Manage Roles: {selectedUser.email}</h3>
                            <button onClick={() => setSelectedUser(null)}><X className="w-5 h-5 text-slate-400" /></button>
                        </div>
                        <div className="p-6 space-y-4">
                            <div className="font-semibold text-slate-700 mb-2">Assigned Roles</div>
                            {selectedUser.assigned_roles && selectedUser.assigned_roles.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {selectedUser.assigned_roles.map(r => (
                                        <div key={r.code} className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm flex items-center gap-2">
                                            {r.name}
                                            <button onClick={() => handleRemove(selectedUser.id, r.code)}>
                                                <X className="w-3 h-3 hover:text-red-600" />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-slate-400 text-sm">No roles assigned.</div>
                            )}

                            <hr className="my-4" />

                            <div className="font-semibold text-slate-700 mb-2">Available Roles</div>
                            <div className="space-y-2">
                                {roles.map(role => {
                                    const isAssigned = selectedUser.assigned_roles?.some(ar => ar.code === role.code);
                                    if (isAssigned) return null;
                                    return (
                                        <div key={role.code} className="flex justify-between items-center p-3 bg-slate-50 rounded-lg border border-slate-100">
                                            <div>
                                                <div className="font-bold text-slate-900">{role.name}</div>
                                                <div className="text-xs text-slate-500">{role.description}</div>
                                            </div>
                                            <button
                                                onClick={() => handleAssign(selectedUser.id, role.code)}
                                                className="p-2 bg-white border border-slate-200 rounded-full hover:bg-indigo-50 hover:text-indigo-600 transition-colors"
                                            >
                                                <Plus className="w-4 h-4" />
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
