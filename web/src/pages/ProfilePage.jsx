import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { useNavigate } from 'react-router-dom'
import { API_URL } from '../../config'
import { TrendingUp, Activity, BarChart2 } from 'lucide-react'

export default function ProfilePage() {
    const [loading, setLoading] = useState(true)
    const [updating, setUpdating] = useState(false)
    const [user, setUser] = useState(null)
    const [profile, setProfile] = useState({
        designation: '',
        organization: '',
        phone: '',
        social_links: { linkedin: '', twitter: '' }
    })
    const [passwordData, setPasswordData] = useState({ newPassword: '', confirmPassword: '' })
    const [passwordError, setPasswordError] = useState('')
    const [notification, setNotification] = useState(null)
    const [accuracyStats, setAccuracyStats] = useState(null)
    const navigate = useNavigate()

    useEffect(() => {
        fetchProfile()
    }, [])

    const fetchProfile = async () => {
        try {
            const { data: { user } } = await supabase.auth.getUser()
            if (!user) {
                navigate('/login')
                return
            }
            setUser(user)

            const { data: profileData, error } = await supabase
                .from('profiles')
                .select('*')
                .eq('id', user.id)
                .single()

            if (data) {
                setProfile({
                    designation: profileData.designation || '',
                    organization: profileData.organization || '',
                    phone: profileData.phone || '',
                    social_links: profileData.social_links || { linkedin: '', twitter: '' }
                })
            }

            // 2. Fetch Accuracy Stats
            const statsRes = await fetch(`${API_URL}/tools/benchmark/stats`, {
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            });
            if (statsRes.ok) {
                setAccuracyStats(await statsRes.json());
            }

        } catch (error) {
            console.error('Error fetching profile:', error)
        } finally {
            setLoading(false)
        }
    }

    const handleUpdateProfile = async (e) => {
        e.preventDefault()
        setUpdating(true)
        setNotification(null)
        try {
            const { error } = await supabase
                .from('profiles')
                .upsert({
                    id: user.id,
                    ...profile,
                    updated_at: new Date()
                })

            if (error) throw error
            setNotification({ type: 'success', message: 'Profile updated successfully!' })
        } catch (error) {
            setNotification({ type: 'error', message: error.message })
        } finally {
            setUpdating(false)
        }
    }

    const handleChangePassword = async (e) => {
        e.preventDefault()
        setPasswordError('')
        if (passwordData.newPassword !== passwordData.confirmPassword) {
            setPasswordError('Passwords do not match')
            return
        }
        if (passwordData.newPassword.length < 6) {
            setPasswordError('Password must be at least 6 characters')
            return
        }

        try {
            const { error } = await supabase.auth.updateUser({ password: passwordData.newPassword })
            if (error) throw error
            setNotification({ type: 'success', message: 'Password changed successfully!' })
            setPasswordData({ newPassword: '', confirmPassword: '' })
        } catch (error) {
            setNotification({ type: 'error', message: error.message })
        }
    }

    const handleDeleteAccount = async () => {
        if (!window.confirm("Are you sure? This action is irreversible and will delete all your data.")) return;

        try {
            // Using our backend API for cleanup which handles public data
            // Since we can't delete Auth User from client easily without function
            // We will clear the profile and sign out.
            const { error } = await supabase.from('profiles').delete().eq('id', user.id)
            if (error) throw error

            await supabase.from('user_credits').delete().eq('user_id', user.id)

            await supabase.auth.signOut()
            navigate('/')
        } catch (error) {
            setNotification({ type: 'error', message: "Could not fully delete account. Please contact support." })
        }
    }

    if (loading) return <div className="min-h-screen bg-slate-50 flex items-center justify-center">Loading...</div>

    return (
        <div className="min-h-screen bg-slate-50 py-12">
            <div className="container mx-auto px-4">
                <div className="max-w-4xl mx-auto">
                    <h1 className="text-3xl font-bold text-slate-900 mb-8 tracking-tight">Account Settings</h1>

                    {notification && (
                        <div className={`p-4 rounded-xl mb-6 ${notification.type === 'success' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>
                            {notification.message}
                        </div>
                    )}

                    <div className="grid gap-8">
                        {/* Profile Info Card */}
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                            <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                                <span>👤</span> Profile Information
                            </h2>
                            <form onSubmit={handleUpdateProfile} className="space-y-6">
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div>
                                        <label className="block text-sm font-bold text-slate-700 mb-2">Email Address</label>
                                        <div className="flex items-center gap-2">
                                            <input
                                                type="email"
                                                value={user?.email}
                                                disabled
                                                className="w-full px-4 py-3 bg-slate-100 border border-slate-200 rounded-xl text-slate-500 cursor-not-allowed"
                                            />
                                            <span className="text-xl" title="Verified">✅</span>
                                        </div>
                                        <p className="text-xs text-slate-400 mt-1">Email cannot be changed.</p>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-bold text-slate-700 mb-2">Phone Number</label>
                                        <input
                                            type="tel"
                                            value={profile.phone}
                                            onChange={e => setProfile({ ...profile, phone: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 transition outline-none"
                                            placeholder="+91..."
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-bold text-slate-700 mb-2">Organization</label>
                                        <input
                                            type="text"
                                            value={profile.organization}
                                            onChange={e => setProfile({ ...profile, organization: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 transition outline-none"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-bold text-slate-700 mb-2">Designation</label>
                                        <div className="relative">
                                            <select
                                                value={profile.designation}
                                                onChange={e => setProfile({ ...profile, designation: e.target.value })}
                                                className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 transition outline-none appearance-none"
                                            >
                                                <option value="Student">Student</option>
                                                <option value="Researcher">Researcher</option>
                                                <option value="Professor">Professor</option>
                                                <option value="Industry">Industry</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <h3 className="text-sm font-bold text-slate-900 mb-4">Social Profiles</h3>
                                    <div className="grid md:grid-cols-2 gap-6">
                                        <div>
                                            <label className="block text-xs font-semibold text-slate-500 mb-1">LinkedIn URL</label>
                                            <input
                                                type="url"
                                                value={profile.social_links.linkedin}
                                                onChange={e => setProfile({ ...profile, social_links: { ...profile.social_links, linkedin: e.target.value } })}
                                                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-1 focus:ring-primary-500 outline-none"
                                                placeholder="https://linkedin.com/in/..."
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-semibold text-slate-500 mb-1">Twitter/X URL</label>
                                            <input
                                                type="url"
                                                value={profile.social_links.twitter}
                                                onChange={e => setProfile({ ...profile, social_links: { ...profile.social_links, twitter: e.target.value } })}
                                                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:ring-1 focus:ring-primary-500 outline-none"
                                                placeholder="https://twitter.com/..."
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div className="flex justify-end">
                                    <button
                                        type="submit"
                                        disabled={updating}
                                        className="btn-primary py-2 px-6 rounded-xl font-bold hover:shadow-lg transition disabled:opacity-50"
                                    >
                                        {updating ? 'Saving...' : 'Save Changes'}
                                    </button>
                                </div>
                            </form>
                        </div>

                        {/* Accuracy Profile Card */}
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                            <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                                <Activity className="w-5 h-5 text-indigo-600" />
                                <span>Accuracy Profile</span>
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
                                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Benchmarks Run</div>
                                    <div className="text-3xl font-black text-slate-800">{accuracyStats?.count || 0}</div>
                                </div>
                                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
                                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Average R²</div>
                                    <div className={`text-3xl font-black ${accuracyStats?.average_r2 > 0.6 ? 'text-emerald-500' : 'text-slate-700'}`}>
                                        {accuracyStats?.average_r2 || '0.00'}
                                    </div>
                                </div>
                                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 text-center">
                                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Peak Accuracy</div>
                                    <div className="text-3xl font-black text-indigo-600">{accuracyStats?.best_r2 || '0.00'}</div>
                                </div>
                            </div>

                            {/* Simple Sparkline */}
                            {accuracyStats?.trend && accuracyStats.trend.length > 0 && (
                                <div className="mt-6">
                                    <div className="text-xs font-bold text-slate-400 uppercase mb-2 flex items-center gap-2">
                                        <TrendingUp className="w-3 h-3" /> Recent Performance Trend
                                    </div>
                                    <div className="h-16 flex items-end gap-1">
                                        {accuracyStats.trend.map((val, i) => (
                                            <div
                                                key={i}
                                                style={{ height: `${(val / (Math.max(...accuracyStats.trend) || 1)) * 100}%` }}
                                                className="flex-1 bg-indigo-200 rounded-t hover:bg-indigo-400 transition-colors relative group"
                                            >
                                                <span className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] bg-slate-800 text-white px-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                                                    {val}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Security Card */}
                        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
                            <h2 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                                <span>🔒</span> Security
                            </h2>
                            <form onSubmit={handleChangePassword} className="space-y-6 max-w-md">
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">New Password</label>
                                    <input
                                        type="password"
                                        value={passwordData.newPassword}
                                        onChange={e => setPasswordData({ ...passwordData, newPassword: e.target.value })}
                                        className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 transition outline-none"
                                        placeholder="••••••••"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-bold text-slate-700 mb-2">Confirm New Password</label>
                                    <input
                                        type="password"
                                        value={passwordData.confirmPassword}
                                        onChange={e => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
                                        className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-primary-500 transition outline-none"
                                        placeholder="••••••••"
                                    />
                                </div>
                                {passwordError && <p className="text-red-500 text-sm">{passwordError}</p>}
                                <button type="submit" className="px-6 py-2 bg-slate-800 text-white font-bold rounded-xl hover:bg-slate-700 transition">
                                    Update Password
                                </button>
                            </form>
                        </div>

                        {/* Danger Zone */}
                        <div className="bg-red-50 p-8 rounded-2xl border border-red-100">
                            <h2 className="text-xl font-bold text-red-700 mb-2">Danger Zone</h2>
                            <p className="text-red-600 mb-6 text-sm">Once you delete your account, there is no going back. Please be certain.</p>
                            <button
                                onClick={handleDeleteAccount}
                                className="px-6 py-2 bg-white text-red-600 font-bold rounded-xl border border-red-200 hover:bg-red-50 transition"
                            >
                                Delete Account Permanently
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
