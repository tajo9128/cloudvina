import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link, useNavigate } from 'react-router-dom'
import JobFilters from '../components/JobFilters'
import { API_URL } from '../config'
import { Activity, Database, Clock, Zap, Search, Filter, ArrowRight, Play, Server, CheckCircle2, FlaskConical } from 'lucide-react'

export default function DashboardPage() {
    const navigate = useNavigate()
    const [user, setUser] = useState(null)
    const [profile, setProfile] = useState(null)
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)
    const [viewMode, setViewMode] = useState('grid') // 'grid' or 'list'
    const [filters, setFilters] = useState({
        status: '',
        search: '',
        minAffinity: '',
        maxAffinity: ''
    })

    // Computed Stats
    const [stats, setStats] = useState({
        active: 0,
        completed: 0,
        avgAffinity: 0,
        totalMolecules: 0
    })

    useEffect(() => {
        // [NEW] First-Time User Check (BioDockify 2.0 Launch)
        const hasVisited = localStorage.getItem('biodockify_visited');
        if (!hasVisited) {
            navigate('/onboarding');
            return;
        }

        fetchUserProfile()
    }, [])

    useEffect(() => {
        fetchJobs()
    }, [filters])

    // [NEW] Real-time Dashboard Updates
    useEffect(() => {
        if (!user) return

        const channel = supabase
            .channel('realtime:dashboard')
            .on(
                'postgres_changes',
                {
                    event: '*',
                    schema: 'public',
                    table: 'jobs',
                    filter: `created_by=eq.${user.id}` // Assuming created_by or user_id column
                },
                () => {
                    fetchJobs()
                }
            )
            .subscribe()

        return () => {
            supabase.removeChannel(channel)
        }
    }, [user])

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
                headers: { 'Authorization': `Bearer ${session.access_token}` }
            })

            if (!response.ok) throw new Error('Failed to fetch jobs')

            const data = await response.json()
            setJobs(data)
            calculateStats(data)

        } catch (error) {
            console.error('Error fetching jobs:', error)
        } finally {
            setLoading(false)
        }
    }

    const calculateStats = (jobData) => {
        const active = jobData.filter(j => ['RUNNING', 'SUBMITTED', 'STARTING'].includes(j.status)).length
        const completed = jobData.filter(j => j.status === 'SUCCEEDED').length

        let totalAffinity = 0
        let affinityCount = 0
        jobData.forEach(j => {
            if (j.binding_affinity) {
                totalAffinity += j.binding_affinity
                affinityCount++
            }
        })
        const avg = affinityCount > 0 ? (totalAffinity / affinityCount).toFixed(2) : 0

        setStats({
            active,
            completed,
            avgAffinity: avg,
            totalMolecules: 100 * active + completed // Rough estimate or real if we had field
        })
    }

    const handleFilterChange = (newFilters) => {
        setFilters(prev => ({ ...prev, ...newFilters }))
    }

    const getStatusColor = (status) => {
        switch (status) {
            case 'SUCCEEDED': return 'bg-emerald-100 text-emerald-700 border-emerald-200'
            case 'FAILED': return 'bg-red-100 text-red-700 border-red-200'
            case 'RUNNING': return 'bg-blue-100 text-blue-700 border-blue-200 animate-pulse'
            default: return 'bg-slate-100 text-slate-700 border-slate-200'
        }
    }

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* Header Section */}
            <div className="bg-white border-b border-slate-200 pt-24 pb-12">
                <div className="container mx-auto px-4">
                    <div className="flex flex-col md:flex-row justify-between items-end gap-6">
                        <div>
                            <h1 className="text-3xl font-bold text-slate-900 mb-2 flex items-center gap-3">
                                <Activity className="w-8 h-8 text-indigo-600" /> Research Dashboard
                            </h1>
                            <p className="text-slate-500">Welcome back, <span className="font-semibold text-slate-700">{profile?.designation || 'Dr.'} {user?.email?.split('@')[0]}</span>. Here's your lab overview.</p>
                        </div>
                        <Link to="/dock/batch" className="btn-primary shadow-lg shadow-primary-500/20 flex items-center gap-2 px-6 py-3 rounded-xl font-bold">
                            <Play className="w-5 h-5 fill-current" />
                            New Campaign
                        </Link>
                    </div>
                </div>
            </div>

            <div className="container mx-auto px-4 -mt-8">

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                    <div className="bg-white rounded-2xl p-6 shadow-xl shadow-slate-200/50 border border-slate-100 flex items-center gap-4">
                        <div className="bg-blue-50 p-4 rounded-xl">
                            <Zap className="w-8 h-8 text-blue-600" />
                        </div>
                        <div>
                            <div className="text-sm font-bold text-slate-400 uppercase tracking-wider">Active Jobs</div>
                            <div className="text-3xl font-bold text-slate-900">{stats.active}</div>
                        </div>
                    </div>
                    <div className="bg-white rounded-2xl p-6 shadow-xl shadow-slate-200/50 border border-slate-100 flex items-center gap-4">
                        <div className="bg-emerald-50 p-4 rounded-xl">
                            <CheckCircle2 className="w-8 h-8 text-emerald-600" />
                        </div>
                        <div>
                            <div className="text-sm font-bold text-slate-400 uppercase tracking-wider">Completed</div>
                            <div className="text-3xl font-bold text-slate-900">{stats.completed}</div>
                        </div>
                    </div>
                    <div className="bg-white rounded-2xl p-6 shadow-xl shadow-slate-200/50 border border-slate-100 flex items-center gap-4">
                        <div className="bg-purple-50 p-4 rounded-xl">
                            <Database className="w-8 h-8 text-purple-600" />
                        </div>
                        <div>
                            <div className="text-sm font-bold text-slate-400 uppercase tracking-wider">Avg Affinity</div>
                            <div className="text-3xl font-bold text-slate-900">{stats.avgAffinity} <span className="text-sm text-slate-400 font-normal">kcal/mol</span></div>
                        </div>
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex flex-col lg:flex-row gap-8">

                    {/* Left: Filter Sidebar (Desktop) or Top (Mobile) */}
                    <aside className="lg:w-64 flex-shrink-0 space-y-6">
                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 sticky top-24">
                            <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                                <Filter className="w-4 h-4" /> Filters
                            </h3>
                            <JobFilters onFilterChange={handleFilterChange} vertical={true} />
                        </div>
                    </aside>

                    {/* Right: Job Grid/List */}
                    <div className="flex-1">

                        {/* View Toggle & Count */}
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold text-slate-800">Recent Implementations</h2>
                            <div className="flex bg-slate-100 p-1 rounded-lg">
                                <button
                                    onClick={() => setViewMode('grid')}
                                    className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${viewMode === 'grid' ? 'bg-white shadow text-slate-900' : 'text-slate-500 hover:text-slate-700'}`}
                                >
                                    Grid
                                </button>
                                <button
                                    onClick={() => setViewMode('list')}
                                    className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${viewMode === 'list' ? 'bg-white shadow text-slate-900' : 'text-slate-500 hover:text-slate-700'}`}
                                >
                                    List
                                </button>
                            </div>
                        </div>

                        {loading ? (
                            <div className="text-center py-20">
                                <div className="inline-block p-4 rounded-full bg-indigo-50 text-indigo-600 mb-4">
                                    <Server className="w-8 h-8 animate-bounce" />
                                </div>
                                <div className="text-slate-500 font-medium">Syncing with cloud cluster...</div>
                            </div>
                        ) : jobs.length === 0 ? (
                            <div className="bg-white rounded-2xl border-2 border-dashed border-slate-300 p-16 text-center">
                                <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6">
                                    <FlaskConical className="w-10 h-10 text-slate-400" />
                                </div>
                                <h3 className="text-xl font-bold text-slate-900 mb-2">Ready to Discover?</h3>
                                <p className="text-slate-500 mb-8 max-w-md mx-auto">Your research pipeline is empty. Launch your first docking campaign to see results.</p>
                                <Link to="/dock/batch" className="btn-primary inline-flex">
                                    Start New Experiment
                                </Link>
                            </div>
                        ) : (
                            <div className={viewMode === 'grid' ? "grid grid-cols-1 md:grid-cols-2 gap-4" : "space-y-4"}>
                                {jobs.map((job) => (
                                    <div key={job.id} className={`group bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-all hover:border-indigo-300 ${viewMode === 'list' ? 'flex items-center justify-between' : ''}`}>

                                        <div className="flex items-start gap-4">
                                            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-indigo-50 to-purple-50 flex items-center justify-center text-indigo-600 font-bold border border-indigo-100 group-hover:scale-105 transition-transform">
                                                {job.ligand_filename ? job.ligand_filename.slice(0, 2).toUpperCase() : 'JG'}
                                            </div>
                                            <div>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <h3 className="font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
                                                        {job.ligand_filename || `Job ${job.id.slice(0, 8)}`}
                                                    </h3>
                                                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold border ${getStatusColor(job.status)}`}>
                                                        {job.status}
                                                    </span>
                                                </div>
                                                <div className="text-xs text-slate-500 flex items-center gap-3">
                                                    <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {new Date(job.created_at).toLocaleDateString()}</span>
                                                    {job.binding_affinity && (
                                                        <span className="font-bold text-slate-700 bg-slate-100 px-1.5 rounded">{job.binding_affinity} kcal/mol</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        <div className={viewMode === 'grid' ? "mt-4 pt-4 border-t border-slate-100 flex justify-between items-center" : ""}>
                                            {viewMode === 'grid' && <span className="text-xs text-slate-400 font-mono">{job.id.slice(0, 8)}</span>}
                                            <Link to={`/dock/${job.id}`} className="text-sm font-bold text-indigo-600 hover:text-indigo-800 flex items-center gap-1 opacity-100 group-hover:translate-x-1 transition-all">
                                                View Analysis <ArrowRight className="w-3 h-3" />
                                            </Link>
                                        </div>

                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

