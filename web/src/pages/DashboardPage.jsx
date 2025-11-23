import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link } from 'react-router-dom'

export default function DashboardPage() {
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetchJobs()
    }, [])

    const fetchJobs = async () => {
        try {
            const { data: { user } } = await supabase.auth.getUser()
            if (!user) return

            const { data, error } = await supabase
                .from('jobs')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false })

            if (error) throw error
            setJobs(data)
        } catch (error) {
            console.error('Error fetching jobs:', error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gray-50">
            <header className="bg-white shadow-sm">
                <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                    <Link to="/" className="flex items-center space-x-2 text-gray-800">
                        <div className="text-2xl">🧬</div>
                        <h1 className="text-xl font-bold">CloudVina</h1>
                    </Link>
                    <div className="flex gap-4">
                        <Link to="/tools/converter" className="text-gray-600 hover:text-purple-600 font-medium">Tools</Link>
                        <button onClick={() => supabase.auth.signOut()} className="text-gray-600 hover:text-red-600">Sign Out</button>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-8">
                <div className="flex justify-between items-center mb-8">
                    <h1 className="text-2xl font-bold text-gray-900">Your Docking Jobs</h1>
                    <Link to="/dock/new" className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition">
                        + New Job
                    </Link>
                </div>

                {loading ? (
                    <div className="text-center py-12">Loading jobs...</div>
                ) : jobs.length === 0 ? (
                    <div className="text-center py-12 bg-white rounded-xl shadow-sm border border-gray-200">
                        <div className="text-4xl mb-4">🧪</div>
                        <h3 className="text-lg font-medium text-gray-900 mb-2">No jobs yet</h3>
                        <p className="text-gray-500 mb-6">Start your first molecular docking job now.</p>
                        <Link to="/dock/new" className="text-purple-600 font-bold hover:text-purple-700">
                            Create Job →
                        </Link>
                    </div>
                ) : (
                    <div className="bg-white rounded-xl shadow-sm overflow-hidden border border-gray-200">
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
            </main>
        </div>
    )
}
