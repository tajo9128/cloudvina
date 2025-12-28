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

    // Auto-refresh every 60 seconds
    useEffect(() => {
        const interval = setInterval(() => {
            fetchJobs();
        }, 60000); // 60 seconds

        return () => clearInterval(interval);
    }, [filter]);

    const fetchJobs = async () => {
        setLoading(true);
        try {
            const { data: { session } } = await supabase.auth.getSession();

            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            let url = `${apiUrl}/admin/jobs?limit=50`;
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
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const response = await fetch(`${apiUrl}/admin/jobs/${jobId}/cancel`, {
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


    const [selectedJob, setSelectedJob] = useState(null);
    const [jobDetails, setJobDetails] = useState(null);
    const [showAuditModal, setShowAuditModal] = useState(false);

    const handleAuditJob = async (job) => {
        setSelectedJob(job);
        setShowAuditModal(true);
        setJobDetails(null);
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const apiUrl = import.meta.env.VITE_API_URL || 'https://cloudvina-api.onrender.com';
            const res = await fetch(`${apiUrl}/admin/jobs/${job.id}/details`, {
                headers: { 'Authorization': `Bearer ${session?.access_token}` }
            });
            if (res.ok) setJobDetails(await res.json());
        } catch (e) { console.error(e); }
    };

    return (
        <div className="p-8">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
                        <Database className="text-primary-400" /> Process Run Audit
                    </h1>
                    <p className="text-slate-400">Monitor, audit, and control system processes.</p>
                </div>
                <button
                    onClick={fetchJobs}
                    className="flex items-center gap-2 px-4 py-2 bg-primary-600/20 text-primary-400 border border-primary-500/30 rounded-lg hover:bg-primary-600/30 transition-colors"
                >
                    <RefreshCw size={16} />Refresh
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
                        className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-primary-500/20 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/30 transition-all font-mono text-sm"
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
            <div className="bg-slate-800/30 border border-primary-500/20 rounded-xl overflow-hidden backdrop-blur-sm shadow-xl">
                <div className="overflow-x-auto">
                    <table className="w-full text-left">
                        <thead>
                            <tr className="bg-slate-800/80 border-b border-primary-500/20 text-slate-400 text-sm uppercase tracking-wider">
                                <th className="p-4 font-bold">Job ID</th>
                                <th className="p-4 font-bold">User</th>
                                <th className="p-4 font-bold">Status</th>
                                <th className="p-4 font-bold">Time</th>
                                <th className="p-4 font-bold">Engine</th>
                                <th className="p-4 text-right font-bold">Control</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-primary-500/10">
                            {loading ? (
                                <tr>
                                    <td colSpan="6" className="p-8 text-center text-slate-400">
                                        <div className="flex items-center justify-center gap-2">
                                            <RefreshCw className="animate-spin" size={16} /> Retrieving Processes...
                                        </div>
                                    </td>
                                </tr>
                            ) : filteredJobs.length === 0 ? (
                                <tr><td colSpan="6" className="p-8 text-center text-slate-400">No active processes found</td></tr>
                            ) : (
                                filteredJobs.map(job => (
                                    <tr key={job.id} className="hover:bg-primary-500/5 transition-colors group">
                                        <td className="p-4">
                                            <button onClick={() => handleAuditJob(job)} className="font-mono text-xs text-primary-300 bg-primary-500/10 px-2 py-1 rounded hover:bg-primary-500/20 transition-colors">
                                                {job.id?.substring(0, 8)}...
                                            </button>
                                        </td>
                                        <td className="p-4">
                                            <div className="text-sm text-white font-medium">{job.profiles?.email || 'System'}</div>
                                            <div className="text-xs text-slate-500">{job.user_id?.substring(0, 6)}...</div>
                                        </td>
                                        <td className="p-4">
                                            {getStatusBadge(job.status)}
                                            {job.error_message && (
                                                <div className="text-red-400 text-xs mt-1 max-w-[200px] truncate" title={job.error_message}>
                                                    {job.error_message}
                                                </div>
                                            )}
                                        </td>
                                        <td className="p-4 text-slate-400 text-xs font-mono">
                                            {new Date(job.created_at).toLocaleString()}
                                        </td>
                                        <td className="p-4 text-slate-300 text-xs uppercase font-bold tracking-wide">
                                            {job.engine || 'VINA'}
                                        </td>
                                        <td className="p-4 text-right">
                                            <div className="flex justify-end gap-2">
                                                <button onClick={() => handleAuditJob(job)} className="text-indigo-400 hover:text-indigo-300 text-xs font-medium bg-indigo-500/10 px-3 py-1 rounded border border-indigo-500/30">
                                                    Audit
                                                </button>
                                                {(job.status === 'running' || job.status === 'queued') && (
                                                    <button
                                                        onClick={() => handleCancelJob(job.id)}
                                                        className="text-red-400 hover:text-red-300 text-xs font-medium bg-red-500/10 px-3 py-1 rounded border border-red-500/30 hover:bg-red-500/20 transition-colors"
                                                    >
                                                        Kill Process
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

            {/* Audit Modal */}
            {showAuditModal && selectedJob && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
                    <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl h-[80vh] flex flex-col shadow-2xl overflow-hidden">
                        <div className="flex justify-between items-center p-6 border-b border-slate-700 bg-slate-800">
                            <div>
                                <h3 className="text-xl font-bold text-white flex items-center gap-3">
                                    Process Audit
                                    <span className="text-sm font-mono font-normal text-slate-400 bg-slate-900 px-2 py-1 rounded">{selectedJob.id}</span>
                                </h3>
                                <p className="text-slate-400 text-sm mt-1">Full context inspection</p>
                            </div>
                            <button onClick={() => setShowAuditModal(false)} className="px-3 py-1 bg-slate-700 hover:bg-white/10 rounded-lg text-white">Close</button>
                        </div>

                        <div className="flex-1 overflow-y-auto p-6 space-y-6">
                            {/* Metadata Grid */}
                            <div className="grid grid-cols-3 gap-4">
                                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                                    <h4 className="text-xs uppercase text-slate-500 font-bold mb-2">Process State</h4>
                                    <div className="text-lg text-white font-mono">{selectedJob.status}</div>
                                </div>
                                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                                    <h4 className="text-xs uppercase text-slate-500 font-bold mb-2">Execution Engine</h4>
                                    <div className="text-lg text-white font-mono">{selectedJob.engine || "Standard"}</div>
                                </div>
                                <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
                                    <h4 className="text-xs uppercase text-slate-500 font-bold mb-2">AWS Batch ID</h4>
                                    <div className="text-sm text-slate-300 font-mono break-all">{selectedJob.batch_job_id || "Local/Pending"}</div>
                                </div>
                            </div>

                            {/* Configuration Dump */}
                            <div className="space-y-2">
                                <h4 className="text-sm font-bold text-white">Input Configuration (S3)</h4>
                                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-green-400 overflow-x-auto">
                                    {jobDetails?.s3_config ? (
                                        <pre>{JSON.stringify(jobDetails.s3_config, null, 2)}</pre>
                                    ) : (
                                        <span className="text-slate-600 italic">No configuration file found in S3 (or access denied).</span>
                                    )}
                                </div>
                            </div>

                            {/* Full DB Record */}
                            <div className="space-y-2">
                                <h4 className="text-sm font-bold text-white">Database Snapshot</h4>
                                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-blue-300 overflow-x-auto">
                                    <pre>{JSON.stringify(selectedJob, null, 2)}</pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
};

export default Jobs;
