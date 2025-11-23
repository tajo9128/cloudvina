import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { supabase } from '../supabaseClient'

export default function AdminPage() {
    const [activeTab, setActiveTab] = useState('users')
    const [loading, setLoading] = useState(false)
    const [users, setUsers] = useState([])
    const [pricingPlans, setpricingPlans] = useState([])
    const [stats, setStats] = useState(null)
    const [error, setError] = useState(null)

    useEffect(() => {
        if (activeTab === 'users') fetchUsers()
        else if (activeTab === 'pricing') fetchPricing()
        else if (activeTab === 'stats') fetchStats()
    }, [activeTab])

    const getAuthHeaders = async () => {
        const { data: { session } } = await supabase.auth.getSession()
        return {
            'Authorization': `Bearer ${session.access_token}`,
            'Content-Type': 'application/json'
        }
    }

    const fetchUsers = async () => {
        setLoading(true)
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const headers = await getAuthHeaders()
            const res = await fetch(`${apiUrl}/admin/users`, { headers })
            if (!res.ok) throw new Error('Failed to fetch users')
            const data = await res.json()
            setUsers(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchPricing = async () => {
        setLoading(true)
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const headers = await getAuthHeaders()
            const res = await fetch(`${apiUrl}/admin/pricing`, { headers })
            if (!res.ok) throw new Error('Failed to fetch pricing')
            const data = await res.json()
            setpricingPlans(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const fetchStats = async () => {
        setLoading(true)
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const headers = await getAuthHeaders()
            const res = await fetch(`${apiUrl}/admin/stats`, { headers })
            if (!res.ok) throw new Error('Failed to fetch stats')
            const data = await res.json()
            setStats(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const toggleVerify = async (userId, currentStatus) => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const headers = await getAuthHeaders()
            const res = await fetch(`${apiUrl}/admin/users/${userId}/verify?verified=${!currentStatus}`, {
                method: 'POST',
                headers
            })
            if (!res.ok) throw new Error('Failed to update user')
            fetchUsers()
        } catch (err) {
            alert(err.message)
        }
    }

    const updateUserRole = async (userId, newRole) => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const headers = await getAuthHeaders()
            const res = await fetch(`${apiUrl}/admin/users/${userId}/role?role=${newRole}`, {
                method: 'POST',
                headers
            })
            if (!res.ok) throw new Error('Failed to update role')
            fetchUsers()
        } catch (err) {
            alert(err.message)
        }
    }

    const updatePlanPrice = async (planId, newPrice) => {
        const plan = pricingPlans.find(p => p.id === planId)
        if (!plan) return

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
            const headers = await getAuthHeaders()
            const res = await fetch(`${apiUrl}/admin/pricing/${planId}`, {
                method: 'PUT',
                headers,
                body: JSON.stringify({
                    ...plan,
                    price: parseFloat(newPrice)
                })
            })
            if (!res.ok) throw new Error('Failed to update pricing')
            fetchPricing()
            {
                ['users', 'pricing', 'stats'].map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${activeTab === tab
                            ? 'border-purple-600 text-purple-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                    >
                        {tab}
                    </button>
                ))
            }
                        </nav >
                    </div >

        <div className="p-6">
            {error && (
                <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-4">
                    {error}
                </div>
            )}

            {loading ? (
                <div className="text-center py-12">Loading...</div>
            ) : (
                <>
                    {/* Users Tab */}
                    {activeTab === 'users' && (
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Verified</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                                        <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {users.map((user) => (
                                        <tr key={user.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{user.email}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                <select
                                                    value={user.role}
                                                    onChange={(e) => updateUserRole(user.id, e.target.value)}
                                                    className="border rounded px-2 py-1"
                                                >
                                                    <option value="user">User</option>
                                                    <option value="admin">Admin</option>
                                                </select>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                <button
                                                    onClick={() => toggleVerify(user.id, user.is_verified)}
                                                    className={`px-3 py-1 rounded-full text-xs font-semibold ${user.is_verified
                                                        ? 'bg-green-100 text-green-800'
                                                        : 'bg-gray-100 text-gray-800'
                                                        }`}
                                                >
                                                    {user.is_verified ? 'Verified' : 'Unverified'}
                                                </button>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {new Date(user.created_at).toLocaleDateString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                                                <button className="text-purple-600 hover:text-purple-900">Manage</button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    {/* Pricing Tab */}
                    {activeTab === 'pricing' && (
                        <div className="space-y-4">
                            {pricingPlans.map((plan) => (
                                <div key={plan.id} className="border border-gray-200 rounded-lg p-6">
                                    <div className="flex justify-between items-start mb-4">
                                        <div>
                                            <h3 className="text-lg font-bold text-gray-900">{plan.name}</h3>
                                            <div className="text-2xl font-bold text-purple-600 mt-1">${plan.price}</div>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <span className={`px-2 py-1 rounded text-xs font-semibold ${plan.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                                                {plan.is_active ? 'Active' : 'Inactive'}
                                            </span>
                                            <button
                                                onClick={() => deletePricingPlan(plan.id)}
                                                className="text-red-600 hover:text-red-800 text-sm"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <div>
                                            <label className="block text-xs text-gray-500">Credits</label>
                                            <input
                                                type="number"
                                                value={plan.credits}
                                                readOnly
                                                className="w-full border-gray-300 rounded-md shadow-sm bg-gray-50"
                                            />
        } finally {
                                                setLoading(false)
                                            }
    }

    const fetchPricing = async () => {
                                                setLoading(true)
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                                            const headers = await getAuthHeaders()
                                            const res = await fetch(`${apiUrl}/admin/pricing`, {headers})
                                            if (!res.ok) throw new Error('Failed to fetch pricing')
                                            const data = await res.json()
                                            setpricingPlans(data)
        } catch (err) {
                                                setError(err.message)
                                            } finally {
                                                setLoading(false)
                                            }
    }

    const fetchStats = async () => {
                                                setLoading(true)
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                                            const headers = await getAuthHeaders()
                                            const res = await fetch(`${apiUrl}/admin/stats`, {headers})
                                            if (!res.ok) throw new Error('Failed to fetch stats')
                                            const data = await res.json()
                                            setStats(data)
        } catch (err) {
                                                setError(err.message)
                                            } finally {
                                                setLoading(false)
                                            }
    }

    const toggleVerify = async (userId, currentStatus) => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                                            const headers = await getAuthHeaders()
                                            const res = await fetch(`${apiUrl}/admin/users/${userId}/verify?verified=${!currentStatus}`, {
                                                method: 'POST',
                                            headers
            })
                                            if (!res.ok) throw new Error('Failed to update user')
                                            fetchUsers()
        } catch (err) {
                                                alert(err.message)
                                            }
    }

    const updateUserRole = async (userId, newRole) => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                                            const headers = await getAuthHeaders()
                                            const res = await fetch(`${apiUrl}/admin/users/${userId}/role?role=${newRole}`, {
                                                method: 'POST',
                                            headers
            })
                                            if (!res.ok) throw new Error('Failed to update role')
                                            fetchUsers()
        } catch (err) {
                                                alert(err.message)
                                            }
    }

    const updatePlanPrice = async (planId, newPrice) => {
        const plan = pricingPlans.find(p => p.id === planId)
                                            if (!plan) return

                                            try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                                            const headers = await getAuthHeaders()
                                            const res = await fetch(`${apiUrl}/admin/pricing/${planId}`, {
                                                method: 'PUT',
                                            headers,
                                            body: JSON.stringify({
                                                ...plan,
                                                price: parseFloat(newPrice)
                })
            })
                                            if (!res.ok) throw new Error('Failed to update pricing')
                                            fetchPricing()
        } catch (err) {
                                                alert(err.message)
                                            }
    }

    const deletePricingPlan = async (planId) => {
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                                            const headers = await getAuthHeaders()
                                            const res = await fetch(`${apiUrl}/admin/pricing/${planId}`, {
                                                method: 'DELETE',
                                            headers
            })
                                            if (!res.ok) throw new Error('Failed to delete pricing plan')
                                            fetchPricing()
        } catch (err) {
                                                alert(err.message)
                                            }
    }
                                            return (
                                            <div className="min-h-screen bg-gray-100 p-8">
                                                <h1 className="text-3xl font-bold text-gray-900 mb-6">Admin Dashboard</h1>

                                                <div className="border-b border-gray-200 mb-6">
                                                    <nav className="-mb-px flex space-x-8">
                                                        {['users', 'pricing', 'stats'].map((tab) => (
                                                            <button
                                                                key={tab}
                                                                onClick={() => setActiveTab(tab)}
                                                                className={`py-4 px-1 border-b-2 font-medium text-sm capitalize ${activeTab === tab
                                                                    ? 'border-purple-600 text-purple-600'
                                                                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                                                    }`}
                                                            >
                                                                {tab}
                                                            </button>
                                                        ))}
                                                    </nav>
                                                </div>

                                                <div className="p-6">
                                                    {error && (
                                                        <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-4">
                                                            {error}
                                                        </div>
                                                    )}

                                                    {loading ? (
                                                        <div className="text-center py-12">Loading...</div>
                                                    ) : (
                                                        <>
                                                            {/* Users Tab */}
                                                            {activeTab === 'users' && (
                                                                <div className="overflow-x-auto">
                                                                    <table className="min-w-full divide-y divide-gray-200">
                                                                        <thead className="bg-gray-50">
                                                                            <tr>
                                                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                                                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                                                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Verified</th>
                                                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                                                                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actions</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="bg-white divide-y divide-gray-200">
                                                                            {users.map((user) => (
                                                                                <tr key={user.id}>
                                                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{user.email}</td>
                                                                                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                                                        <select
                                                                                            value={user.role}
                                                                                            onChange={(e) => updateUserRole(user.id, e.target.value)}
                                                                                            className="border rounded px-2 py-1"
                                                                                        >
                                                                                            <option value="user">User</option>
                                                                                            <option value="admin">Admin</option>
                                                                                        </select>
                                                                                    </td>
                                                                                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                                                        <button
                                                                                            onClick={() => toggleVerify(user.id, user.is_verified)}
                                                                                            className={`px-3 py-1 rounded-full text-xs font-semibold ${user.is_verified
                                                                                                ? 'bg-green-100 text-green-800'
                                                                                                : 'bg-gray-100 text-gray-800'
                                                                                                }`}
                                                                                        >
                                                                                            {user.is_verified ? 'Verified' : 'Unverified'}
                                                                                        </button>
                                                                                    </td>
                                                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                                                        {new Date(user.created_at).toLocaleDateString()}
                                                                                    </td>
                                                                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                                                                                        <button className="text-purple-600 hover:text-purple-900">Manage</button>
                                                                                    </td>
                                                                                </tr>
                                                                            ))}
                                                                        </tbody>
                                                                    </table>
                                                                </div>
                                                            )}

                                                            {/* Pricing Tab */}
                                                            {activeTab === 'pricing' && (
                                                                <div className="space-y-4">
                                                                    {pricingPlans.map((plan) => (
                                                                        <div key={plan.id} className="border border-gray-200 rounded-lg p-6">
                                                                            <div className="flex justify-between items-start mb-4">
                                                                                <div>
                                                                                    <h3 className="text-lg font-bold text-gray-900">{plan.name}</h3>
                                                                                    <div className="text-2xl font-bold text-purple-600 mt-1">${plan.price}</div>
                                                                                </div>
                                                                                <div className="flex items-center space-x-2">
                                                                                    <span className={`px-2 py-1 rounded text-xs font-semibold ${plan.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                                                                                        {plan.is_active ? 'Active' : 'Inactive'}
                                                                                    </span>
                                                                                    <button
                                                                                        onClick={() => deletePricingPlan(plan.id)}
                                                                                        className="text-red-600 hover:text-red-800 text-sm"
                                                                                    >
                                                                                        Delete
                                                                                    </button>
                                                                                </div>
                                                                            </div>

                                                                            <div className="space-y-2">
                                                                                <div>
                                                                                    <label className="block text-xs text-gray-500">Credits</label>
                                                                                    <input
                                                                                        type="number"
                                                                                        value={plan.credits}
                                                                                        readOnly
                                                                                        className="w-full border-gray-300 rounded-md shadow-sm bg-gray-50"
                                                                                    />
                                                                                </div>
                                                                                <div>
                                                                                    <label className="block text-xs text-gray-500">Price ($)</label>
                                                                                    <input
                                                                                        type="number"
                                                                                        value={plan.price}
                                                                                        onChange={(e) => updatePlanPrice(plan.id, e.target.value)}
                                                                                        className="w-full border-gray-300 rounded-md shadow-sm"
                                                                                    />
                                                                                </div>
                                                                                <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                                                                                <div className="text-sm text-gray-500">{stat.label}</div>
                                                                            </div>
                                                                    ))}
                                                                        </div>
                                                                    )}
                                                                </>
                                                            )}
                                                        </div>
                                                </div>
                                                )
}
