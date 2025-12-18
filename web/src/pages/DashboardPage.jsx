import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link } from 'react-router-dom'
import JobFilters from '../components/JobFilters'
import { API_URL } from '../config'

export default function DashboardPage() {
    const [user, setUser] = useState(null)
    const [profile, setProfile] = useState(null)
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)
    const [filters, setFilters] = useState({
        status: '',
        search: '',
        minAffinity: '',
        maxAffinity: ''
    })

    useEffect(() => {
        fetchJobs()
        fetchUserProfile()
    }, [filters])

    const fetchUserProfile = async () => {
        const { data: { user } } = await supabase.auth.getUser()
        if (user) {
            setUser(user)
            const { data } = await supabase.from('profiles').select('*').eq('id', user.id).single()
            if (data) setProfile(data)
        }
    }

    const fetchJobs = async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession()
            if (!session) return

            const params = new URLSearchParams()
            if (filters.status) params.append('status', filters.status)
            if (filters.search) params.append('search', filters.search)
            if (filters.minAffinity) params.append('min_affinity', filters.minAffinity)
            if (filters.maxAffinity) params.append('max_affinity', filters.maxAffinity)

            const url = `${API_URL}/jobs?${params.toString()}`

            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${session.access_token}`
                }
            })

            if (!response.ok) throw new Error('Failed to fetch jobs')

            const data = await response.json()
            setJobs(data)
        } catch (error) {
            console.error('Error fetching jobs:', error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* Header Section */}
            <div className="bg-white border-b border-slate-200 pt-24 pb-12">
                <div className="container mx-auto px-4">
                    <div className="flex justify-between items-end">
                        <div>
                            <h1 className="text-3xl font-bold text-slate-900 mb-2">Dashboard</h1>
                            <p className="text-slate-500">Manage and monitor your molecular docking simulations</p>
                        </div>
                        <Link to="/dock/new" className="btn-primary">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                            New Job
                        </Link>
                    </div>
                </div>
            </div>

            <div className="container mx-auto px-4 py-8">

                {/* User Welcome Card (Fix for "Plain User" issue) */}
                {user && (
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-8 flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-500 to-secondary-500 text-white flex items-center justify-center text-2xl font-bold shadow-md">
                                {user.email[0].toUpperCase()}
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-slate-900">
                                    Welcome back, {profile?.designation ? `${profile.designation} ` : ''}
                                    <span className="text-primary-600">{user.email.split('@')[0]}</span>!
                                </h2>
                                <p className="text-slate-500 text-sm">
                                    {profile?.organization ? `${profile.organization} • ` : ''}
                                    Ready for your next breakthrough?
                                </p>
                            </div>
                        </div>
                        <div className="hidden md:block text-right">
                            <div className="text-sm text-slate-400 uppercase tracking-wider font-bold mb-1">Current Plan</div>
                            <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-bold">
                                Free Tier Active
                            </div>
                        </div>
                    </div>
                )}

                {/* Job Filters */}
                <div className="mb-8">
                    <JobFilters onFilterChange={handleFilterChange} />
                </div>

                {loading ? (
                    <div className="text-center py-20">
                        <div className="inline-block p-4 rounded-full bg-primary-50 text-primary-600 mb-4">
                            <svg className="w-8 h-8 animate-spin" fill="none" viewBox="0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                        </div>
                        <div className="text-slate-500 font-medium">Loading your research data...</div>
                    </div>
                ) : jobs.length === 0 ? (
                    <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center shadow-sm">
                        <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6 text-4xl">🧪</div>
                        <h3 className="text-xl font-bold text-slate-900 mb-2">No jobs found</h3>
                        <p className="text-slate-500 mb-8 max-w-md mx-auto">You haven't run any docking simulations yet. Start your first job to see results here.</p>
                        <Link to="/dock/new" className="btn-primary inline-flex">
                            Create First Job
                        </Link>
                    </div>
                ) : (
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-slate-200">
                                <thead className="bg-slate-50">
                                    <tr>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Job ID</th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Status</th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Created</th>
                                        <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">Affinity</th>
                                        <th className="px-6 py-4 text-right text-xs font-semibold text-slate-500 uppercase tracking-wider">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-200 bg-white">
                                    {jobs.map((job) => (
                                        <tr key={job.id} className="hover:bg-slate-50 transition-colors group">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">
                                                <Link to={`/dock/${job.id}`} className="font-mono text-primary-600 hover:text-primary-700 hover:underline">
                                                    {job.id.slice(0, 8)}
                                                </Link>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full 
                                            ${job.status === 'SUCCEEDED' ? 'bg-green-100 text-green-700 border border-green-200' :
                                                        job.status === 'FAILED' ? 'bg-red-100 text-red-700 border border-red-200' :
                                                            'bg-amber-100 text-amber-700 border border-amber-200'}`}>
                                                    {job.status}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                                                {new Date(job.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
                                                {job.binding_affinity ? (
                                                    <span className="text-slate-900">{job.binding_affinity} <span className="text-slate-400 font-normal text-xs">kcal/mol</span></span>
                                                ) : (
                                                    <span className="text-slate-300">-</span>
                                                )}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                                <Link to={`/dock/${job.id}`} className="text-slate-400 hover:text-primary-600 font-medium transition-colors">
                                                    View Details →
                                                </Link>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
