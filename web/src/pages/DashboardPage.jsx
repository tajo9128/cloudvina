import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link } from 'react-router-dom'
import JobFilters from '../components/JobFilters'
import { API_URL } from '../config'

export default function DashboardPage() {
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
    }, [filters])

    const handleFilterChange = (newFilters) => {
        setFilters(prev => ({ ...prev, ...newFilters }))
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
        <div className="min-h-screen bg-gradient-to-b from-deep-navy-900 to-deep-navy-800">
            <div className="container mx-auto px-4 py-8">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
                        <p className="text-blue-200">Manage your docking jobs</p>
                    </div>
                    <Link to="/dock/new" className="btn-blue-glow">
                        + New Job
                    </Link>
                </div>

                {/* Job Filters */}
                <JobFilters onFilterChange={handleFilterChange} />

                {loading ? (
                    <div className="text-center py-12">
                        <div className="text-white">
                            <div className="text-5xl mb-4">🧪</div>
                            <div className="text-xl">Loading jobs...</div>
                        </div>
                    </div>
                ) : jobs.length === 0 ? (
                    <div className="glass-card-light p-12 text-center">
                        <div className="text-4xl mb-4">🧪</div>
                        <h3 className="text-lg font-medium text-deep-navy-900 mb-2">No jobs yet</h3>
                        <p className="text-gray-600 mb-6">Start your first molecular docking job now.</p>
                        <Link to="/dock/new" className="text-blue-600 font-bold hover:text-blue-700">
                            Create Job →
                        </Link>
                    </div>
                ) : (
                    <div className="glass-card-light overflow-hidden">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Job ID</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Affinity</th>
                                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {jobs.map((job) => (
                                    <tr key={job.id} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                            <Link to={`/dock/${job.id}`} className="hover:text-purple-600">
                                                {job.id.slice(0, 8)}...
                                            </Link>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                        ${job.status === 'SUCCEEDED' ? 'bg-green-100 text-green-800' :
                                                    job.status === 'FAILED' ? 'bg-red-100 text-red-800' :
                                                        'bg-yellow-100 text-yellow-800'}`}>
                                                {job.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {new Date(job.created_at).toLocaleDateString()}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-bold">
                                            {job.binding_affinity ? `${job.binding_affinity} kcal/mol` : '-'}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                            <Link to={`/dock/${job.id}`} className="text-purple-600 hover:text-purple-900">
                                                View
                                            </Link>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    )
}
