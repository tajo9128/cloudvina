import React, { useState, useEffect } from 'react';
import { supabase } from '../../supabaseClient';
// import { Search, XCircle, CheckCircle, Clock, AlertCircle, RefreshCw, Database } from 'lucide-react';

const Search = () => <span>🔍</span>;
const XCircle = () => <span>❌</span>;
const CheckCircle = () => <span>✅</span>;
const Clock = () => <span>🕒</span>;
const AlertCircle = () => <span>⚠️</span>;
const RefreshCw = () => <span>🔄</span>;
const Database = () => <span>💾</span>;

const Jobs = () => {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all');
    const [search, setSearch] = useState('');

    useEffect(() => {
        fetchJobs();
    }, [filter]);

    const fetchJobs = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();

            let url = `${import.meta.env.VITE_API_URL}/admin/jobs?limit=50`;
            if (filter !== 'all') {
                url += `&status=${filter}`;
            }

            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setJobs(data);
            }
        } catch (error) {
            console.error('Error fetching jobs:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCancelJob = async (jobId) => {
        if (!confirm('Are you sure you want to cancel this job?')) return;

        try {
            const { data: { session } } = await supabase.auth.getSession();
            const response = await fetch(`${import.meta.env.VITE_API_URL}/admin/jobs/${jobId}/cancel`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session?.access_token}`
                }
            });

            if (response.ok) {
                alert('Job cancelled successfully');
                fetchJobs();
            } else {
                alert('Failed to cancel job');
            }
        } catch (error) {
            console.error('Error cancelling job:', error);
        }
    };

    const getStatusBadge = (status) => {
        const styles = {
            completed: 'bg-green-500/10 text-green-400 border-green-500/30',
            failed: 'bg-red-500/10 text-red-400 border-red-500/30',
            running: 'bg-primary-500/10 text-primary-400 border-primary-500/30',
            queued: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
            cancelled: 'bg-slate-500/10 text-slate-400 border-slate-500/30',
        };

        const icons = {
            completed: <CheckCircle size={12} />,
            failed: <XCircle size={12} />,
            running: <RefreshCw size={12} className="animate-spin" />,
            queued: <Clock size={12} />,
            cancelled: <XCircle size={12} />,
        };

        return (
            <span className={`px-2 py-1 rounded-full text-xs font-medium border flex items-center gap-1 w-fit ${styles[status] || styles.cancelled}`}>
                {icons[status]}
                {status}
            </span>
        );
    };

    const filteredJobs = jobs.filter(job =>
        job.id?.includes(search) ||
        job.profiles?.email?.includes(search) ||
        job.receptor_filename?.includes(search)
    );

    return (
        <div className="p-8">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
                        <Database className="text-primary-400" /> Job Management
                    </h1>
                    <p className="text-slate-400">Monitor and control docking jobs</p>
                </div>
                <button
                    onClick={fetchJobs}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-600/20 text-primary-400 border border-primary-500/30 rounded-lg hover:bg-primary-600/30 transition-colors"
                >
                    <RefreshCw size={16} />
                    Refresh
                </button>
            </header>

            {/* Filters & Search */}
            <div className="flex flex-col md:flex-row gap-4 mb-6">
                <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
                    <input
                        type="text"
                        placeholder="Search by Job ID, User Email, or Receptor..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-primary-500/20 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/30 transition-all"
                    />
                </div>
                <div className="flex gap-2 flex-wrap">
                    {['all', 'running', 'queued', 'failed', 'completed'].map(status => (
                        <button
                            key={status}
                            onClick={() => setFilter(status)}
                            className={`px-4 py-2 rounded-lg capitalize transition-all duration-200 ${filter === status
                                ? 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-lg shadow-primary-500/20'
                                : 'bg-slate-800/50 text-slate-400 border border-primary-500/20 hover:bg-primary-500/10 hover:text-primary-300'
                                }`}
                        >
                            {status}
                        </button>
                    ))}
                </div>
            </div>

            {/* Jobs Table */}
            <div className="bg-slate-800/30 border border-primary-500/20 rounded-xl overflow-hidden backdrop-blur-sm">
                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead>
                            <tr className="bg-slate-800/50 border-b border-primary-500/20 text-slate-400 text-sm">
                                <th className="p-4 font-medium">Job ID</th>
                                <th className="p-4 font-medium">User</th>
                                <th className="p-4 font-medium">Status</th>
                                <th className="p-4 font-medium">Created At</th>
                                <th className="p-4 font-medium">Target</th>
                                <th className="p-4 text-right font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-primary-500/10">
                            {loading ? (
                                <tr>
                                    <td colSpan="6" className="p-8 text-center text-slate-400">
                                        <div className="flex items-center justify-center gap-2">
                                            <RefreshCw className="animate-spin" size={16} />
                                            Loading jobs...
                                        </div>
                                    </td>
                                </tr>
                            ) : filteredJobs.length === 0 ? (
                                <tr>
                                    <td colSpan="6" className="p-8 text-center text-slate-400">No jobs found</td>
                                </tr>
                            ) : (
                                filteredJobs.map(job => (
                                    <tr key={job.id} className="hover:bg-primary-500/5 transition-colors">
                                        <td className="p-4">
                                            <span className="font-mono text-sm text-primary-300 bg-primary-500/10 px-2 py-1 rounded" title={job.id}>
                                                {job.id?.substring(0, 8)}...
                                            </span>
                                        </td>
                                        <td className="p-4">
                                            <div className="text-sm text-white">{job.profiles?.email || 'Unknown'}</div>
                                        </td>
                                        <td className="p-4">
                                            {getStatusBadge(job.status)}
                                        </td>
                                        <td className="p-4 text-slate-400 text-sm">
                                            {new Date(job.created_at).toLocaleString()}
                                        </td>
                                        <td className="p-4 text-slate-300 text-sm font-mono">
                                            {job.receptor_filename || 'N/A'}
                                        </td>
                                        <td className="p-4 text-right">
                                            {(job.status === 'running' || job.status === 'queued') && (
                                                <button
                                                    onClick={() => handleCancelJob(job.id)}
                                                    className="text-red-400 hover:text-red-300 text-sm font-medium flex items-center gap-1 ml-auto bg-red-500/10 px-3 py-1 rounded-lg border border-red-500/30 hover:bg-red-500/20 transition-colors"
                                                >
                                                    <XCircle size={14} /> Cancel
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

export default Jobs;
