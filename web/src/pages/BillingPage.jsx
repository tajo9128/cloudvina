import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'
import { Link } from 'react-router-dom'

export default function BillingPage() {
    const [loading, setLoading] = useState(true)
    const [credits, setCredits] = useState(null)
    const [history, setHistory] = useState([])

    useEffect(() => {
        fetchBillingData()
    }, [])

    const fetchBillingData = async () => {
        try {
            const { data: { user } } = await supabase.auth.getUser()
            if (!user) return

            // 1. Fetch Credits
            const { data: creditsData } = await supabase.rpc('get_available_credits', { p_user_id: user.id })
            if (creditsData && creditsData.length > 0) {
                setCredits(creditsData[0])
            }

            // 2. Fetch Usage History (Jobs as proxy)
            const { data: jobsData } = await supabase
                .from('jobs')
                .select('id, created_at, job_type, protein_name, status')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false })
                .limit(20)

            setHistory(jobsData || [])

        } catch (error) {
            console.error("Billing fetch error:", error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) return <div className="min-h-screen bg-slate-50 flex items-center justify-center">Loading...</div>

    return (
        <div className="min-h-screen bg-slate-50 py-12">
            <div className="container mx-auto px-4">
                <div className="max-w-5xl mx-auto">
                    <h1 className="text-3xl font-bold text-slate-900 mb-8 tracking-tight">Billing & Credits</h1>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <p className="text-sm font-bold text-slate-500 mb-1">Total Credits</p>
                            <p className="text-3xl font-black text-primary-600">{credits?.total_credits || 0}</p>
                        </div>
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <p className="text-sm font-bold text-slate-500 mb-1">Monthly Plan</p>
                            <p className="text-xl font-bold text-slate-800 capitalize">{credits?.plan || 'Free'}</p>
                            <p className="text-xs text-slate-400 mt-1">Resets monthly</p>
                        </div>
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <p className="text-sm font-bold text-slate-500 mb-1">Bonus Credits</p>
                            <p className="text-2xl font-bold text-slate-800">{credits?.bonus_credits || 0}</p>
                        </div>
                        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <p className="text-sm font-bold text-slate-500 mb-1">Purchased</p>
                            <p className="text-2xl font-bold text-slate-800">{credits?.paid_credits || 0}</p>
                        </div>
                    </div>

                    {/* Buy Credits Banner */}
                    <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white rounded-3xl p-8 mb-12 shadow-xl relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-64 h-64 bg-white opacity-5 rounded-full -mr-16 -mt-16 blur-3xl"></div>
                        <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-6">
                            <div>
                                <h2 className="text-2xl font-bold mb-2">Need more computing power?</h2>
                                <p className="text-slate-300 max-w-lg">
                                    Purchase additional credits to run large-scale batch docking jobs or complex MD simulations.
                                    <span className="block mt-2 font-bold text-white">Rate: ₹1 = 1 Credit</span>
                                </p>
                            </div>
                            <Link to="/contact" className="px-8 py-3 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-xl transition shadow-lg shadow-primary-900/50">
                                Contact Sales to Buy
                            </Link>
                        </div>
                    </div>

                    {/* Usage History */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                        <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                            <h3 className="text-lg font-bold text-slate-900">Recent Usage History</h3>
                            <Link to="/dock/new" className="text-primary-600 text-sm font-bold hover:underline">New Job →</Link>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left">
                                <thead className="bg-slate-50 border-b border-slate-100">
                                    <tr>
                                        <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase">Date</th>
                                        <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase">Activity Details</th>
                                        <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase text-right">Cost</th>
                                        <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase text-center">Status</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                    {history.length === 0 ? (
                                        <tr>
                                            <td colSpan="4" className="px-6 py-8 text-center text-slate-500">
                                                No usage history found.
                                            </td>
                                        </tr>
                                    ) : (
                                        history.map((job) => (
                                            <tr key={job.id} className="hover:bg-slate-50/50 transition">
                                                <td className="px-6 py-4 text-sm text-slate-600">
                                                    {new Date(job.created_at).toLocaleDateString()}
                                                </td>
                                                <td className="px-6 py-4">
                                                    <p className="font-bold text-slate-800 text-sm">
                                                        {job.job_type === 'docking' ? 'Molecular Docking' : 'Batch Processing'}
                                                    </p>
                                                    <p className="text-xs text-slate-500 truncate max-w-[200px]">
                                                        {job.protein_name || 'Unknown Target'}
                                                    </p>
                                                </td>
                                                <td className="px-6 py-4 text-right">
                                                    <span className="inline-block px-2 py-1 bg-red-50 text-red-600 rounded text-xs font-bold">
                                                        -1 Credit
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-center">
                                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${job.status === 'completed' ? 'bg-green-100 text-green-800' :
                                                            job.status === 'failed' ? 'bg-red-100 text-red-800' :
                                                                'bg-blue-100 text-blue-800'
                                                        }`}>
                                                        {job.status}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    )
}
